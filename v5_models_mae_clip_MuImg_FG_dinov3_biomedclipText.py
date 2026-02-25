import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from functools import partial
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from dinov3.models.vision_transformer import vit_base, vit_large
import open_clip
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block

def build_M2GCLIP(model_type, args):
    assert model_type in ['vit_base_patch16', 'vit_large_patch16']
    # biomedclip, _ = open_clip.create_model_from_pretrained(
    #     'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    # )
    biomedclip, _ = open_clip.create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir=os.path.expanduser('~/.cache/huggingface/hub')
        # "/mnt/data/yizhenyu/hf_models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    if model_type == 'vit_base_patch16':
        dino_backbone = vit_base(
            patch_size=16,
            n_storage_tokens=4,       # 解决 encoder.storage_tokens 意外
            layerscale_init=1e-5,     # 解决 ls1.gamma, ls2.gamma 意外
            mask_k_bias=True,         # 解决 bias_mask 意外
        )
        dino_weight_path='/mnt/data/xujianwei.xjw/workspace/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        # load_mae_pretrain(dino_backbone, '/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/pretrained/mae_pretrain_vit_base.pth')
        model = MaskedAutoencoder_MuImg_ViT(dino_backbone, biomedclip,
            in_chans=3, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=args.norm_pix_loss,
            )
        if not args.train_from_scratch:
            load_dinov3_weights(model.encoder, dino_weight_path)
    elif model_type == 'vit_large_patch16':
        dino_backbone = vit_large(
            patch_size=16,
            n_storage_tokens=4,       
            layerscale_init=1e-5,     
            mask_k_bias=True,         
        )
        dino_weight_path='/mnt/data/xujianwei.xjw/workspace/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        model = MaskedAutoencoder_MuImg_ViT(dino_backbone, biomedclip,
            in_chans=3, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=args.norm_pix_loss,
            )
        if not args.train_from_scratch:
            load_dinov3_weights(model.encoder, dino_weight_path)
    load_biomedclip_weights(model, biomedclip)
    return model

def load_biomedclip_weights(model, biomedclip):
    model.text_encoder.load_state_dict(biomedclip.text.state_dict())

def load_dinov3_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k.replace('backbone.', '')] = v
        else:
            new_state_dict[f'{k}'] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded DinoV3 weights from {checkpoint_path}")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    # print(f"Unexpected keys: {msg.unexpected_keys}")
    # print(f"Missing keys: {msg.missing_keys}")
    return msg


def load_mae_pretrain(dino_backbone, checkpoint_path):
    """
    加载 MAE 官方预训练权重 (例如 mae_pretrain_vit_base.pth)
    官方权重的 Key 与当前类成员变量名一致，无需复杂映射。
    """
    import os
    if not os.path.exists(checkpoint_path):
        print(f"Warning: MAE checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # MAE 官方权重通常打包在 'model' key 下
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 1. 过滤掉权重文件中的 Decoder 部分和 Mask Token
    # 官方权重包含：patch_embed, pos_embed, blocks, norm (Encoder) 
    # 以及 decoder_xxx, mask_token (Decoder)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 我们只加载 Encoder 需要的部分，排除 Decoder 和权重文件里的 mask_token
        if k.startswith('decoder') or k == 'mask_token':
            continue
        
        # 检查当前模型是否含有这个 key
        if k in dino_backbone.state_dict():
            # 检查形状是否匹配 (防止 Base 加载到 Large 上)
            if v.shape == dino_backbone.state_dict()[k].shape:
                new_state_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: skipping...")

    # 2. 执行加载
    # 使用 strict=False，因为我们故意漏掉了 Decoder 的权重
    msg = dino_backbone.load_state_dict(new_state_dict, strict=False)
    
    print(f"--- MAE Official Weights Loaded ---")
    print(f"Path: {checkpoint_path}")
    print(f"Matched Keys: {len(new_state_dict)}")
    
    # 正常的 Missing Keys 应该只包含 decoder 相关的层和当前模型特有的变量
    missing_encoder_keys = [k for k in msg.missing_keys if not k.startswith('decoder')]
    if missing_encoder_keys:
        print(f"Missing Encoder Keys: {missing_encoder_keys}")
    else:
        print("Encoder weights matched perfectly!")
        
    return msg

def all_gather_with_variable_size(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather tensors whose first dimension (length) may differ across ranks.

    This function gathers `tensor` from all ranks, where each rank may have a different
    `tensor.size(0)`. It pads each local tensor to the maximum length across ranks,
    performs `dist.all_gather` on the padded tensors, then removes padding and concatenates
    the valid parts along dim=0.

    Notes:
        - This implementation uses `dist.all_gather` and is NOT autograd-differentiable.
          (Suitable for gathering detached features / building a global memory bank.)

    Args:
        tensor: A tensor of shape [N_local, ...].

    Returns:
        A concatenated tensor of shape [sum_i N_i, ...] containing all ranks' valid data.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    device = tensor.device

    # 1) Gather local lengths from all ranks
    local_len = torch.tensor([tensor.size(0)], device=device, dtype=torch.long)  # shape [1]
    len_list = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(len_list, local_len)
    lengths = [int(x.item()) for x in len_list]
    max_len = max(lengths)

    # 2) Pad local tensor to max_len along dim=0
    padded = tensor
    if tensor.size(0) < max_len:
        pad_shape = (max_len - tensor.size(0),) + tensor.shape[1:]
        padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)

    # 3) All-gather padded tensors
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # 4) Remove padding and concatenate valid parts
    out = torch.cat([g[:l] for g, l in zip(gathered, lengths)], dim=0)
    return out


def all_gather_features(features):
    """
    Collect features from all GPU nodes with gradient support.
    """
    if not dist.is_initialized():
        return features
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Create a list to store gathered tensors
    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    
    # Perform all_gather (Standard DDP way)
    dist.all_gather(gathered_features, features)
    
    # To allow backpropagation to other GPUs' features (Optional for CLIP, but standard for some)
    # However, standard CLIP only computes gradients for the local features against global ones.
    # We replace the local part of the gathered list with the original features to keep the graph.
    gathered_features[rank] = features
    
    return torch.cat(gathered_features, dim=0)

class FG_ProjectionHead(nn.Module):
    """Feature head to be used for feature fusion.

    :param config: Configuration for BERT.
    """

    def __init__(self, input_size, output_size, hidden_size=512) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(input_size, hidden_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dense_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(x)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        out = self.dense_to_output(hidden_states)

        return out


class MaskedAutoencoder_MuImg_ViT(nn.Module):
    '''
    Masked Autoencoder for Multi-Image -- Text Interaction
    '''
    def __init__(self, dino_backbone, biomedclip, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, train_from_scratch=False, dino_weight_path=None):
        super().__init__()

        # --- Dino Encoder ---
        self.encoder = dino_backbone
        self.encoder.mask_token.requires_grad = False # Freeze mask token since we use our own mae strategy

        self.embed_dim = dino_backbone.embed_dim
        self.patch_size = dino_backbone.patch_size
        self.num_patches = self.encoder.patch_embed.num_patches

        # --- MAE Decoder ---
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * in_chans, bias=True)

        # --- CLIP Text Encoder ---
        self.text_encoder = biomedclip.text
        text_embed_dim = biomedclip.text.proj[-1].out_features if hasattr(biomedclip.text, 'proj') else 512

        # Unfreeze text branch by default
        # self.text_encoder.requires_grad_(False)

        # --- Image Projection for cls_token & path_tokens aggregation ---
        self.image_feat_projection = nn.Linear(self.embed_dim*2, self.embed_dim)

        # --- Image Projection for CLIP ---
        # self.text_projection = biomedclip.text.proj
        self.image_projection = nn.Linear(self.embed_dim, text_embed_dim, bias=False)
        # self.logit_scale = nn.Parameter(biomedclip.logit_scale.clone())
        # if train_from_scratch:
        #     self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # else:
        #       self.logit_scale = nn.Parameter(biomedclip.logit_scale.clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # --- Image Projection for FG-CLIP ---
        self.image_projection_fg = FG_ProjectionHead(self.embed_dim, text_embed_dim) 
        # self.text_projection_fg = FG_ProjectionHead(text_embed_dim, text_embed_dim) 
        self.logit_scale_fg = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.norm_pix_loss = norm_pix_loss

    def initialize_weights(self):
        # Sine-cos pos_embed initialization
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Basic init
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

        if hasattr(self.encoder, 'init_weights'):
            self.encoder.init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        if mask_ratio == 0.0:
            return x, torch.zeros([N, L], device=x.device), torch.arange(L, device=x.device).repeat(N, 1), torch.arange(L, device=x.device).repeat(N, 1)
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):

        if mask_ratio == 0:
            dino_outputs = self.encoder.forward_features(x)
            # 从 dino 输出的字典中提取并拼接，以原有的 tokens 结构: [CLS, Storage, Patches]
            # dino_outputs 包含: x_norm_clstoken, x_storage_tokens, x_norm_patchtokens
            cls_token_out = dino_outputs["x_norm_clstoken"].unsqueeze(1) # [B, 1, D]
            storage_tokens_out = dino_outputs["x_storage_tokens"]       # [B, n_stg, D]
            patch_tokens_out = dino_outputs["x_norm_patchtokens"]       # [B, L, D]
            
            tokens = torch.cat([cls_token_out, storage_tokens_out, patch_tokens_out], dim=1)
            return tokens, None, None

        # 1. Patch Embed
        # 输入 x: [B, 3, 224, 224] -> 输出: [B, 14, 14, 768]
        x = self.encoder.patch_embed(x) 
        B, H, W, D = x.shape
        x = x.flatten(1, 2) # [B, 196, 768]

        # 2. Random Masking (MAE 核心逻辑)
        # x_masked: [B, L_keep, D], 例如 75% mask 则 L_keep = 49
        # ids_keep: [B, L_keep]
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # 3. Prepare Tokens (CLS + Storage + Masked Patches)
        # a. CLS Token: [B, 1, 768]
        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        
        # b. Storage Tokens (Registers): [B, n_storage, 768]
        # 根据报错，DinoV3 默认通常有 4 个 storage tokens
        if self.encoder.n_storage_tokens > 0:
            storage_tokens = self.encoder.storage_tokens.expand(B, -1, -1)
            # 拼接: [B, 1 + n_storage + L_keep, 768]
            tokens = torch.cat([cls_token, storage_tokens, x_masked], dim=1)
        else:
            # 拼接: [B, 1 + L_keep, 768]
            tokens = torch.cat([cls_token, x_masked], dim=1)

        # 4. RoPE 处理 (仅针对 Patch 部分进行 Gather)
        # 获取原始 2D 坐标对应的 cos/sin: 两个 [196, 64]
        rope_full = self.encoder.rope_embed(H=H, W=W) 
        d_h = rope_full[0].shape[1] # 64

        # 构造用于 gather 的索引: [B, L_keep, 64]
        # 注意：这里只针对保留下来的 patch，不包含 cls 和 storage
        ids_keep_rope = ids_keep.unsqueeze(-1).expand(-1, -1, d_h)

        rope_final_list = []
        for r_tensor in rope_full:
            # a. 扩展 B 维度并根据索引选出对应的位置编码
            # [1, 196, 64] -> expand -> [B, 196, 64] -> gather -> [B, L_keep, 64]
            r_expanded = r_tensor.unsqueeze(0).expand(B, -1, -1)
            r_selected = torch.gather(r_expanded, dim=1, index=ids_keep_rope)
            
            # b. 【重要】：DinoV3 的 Block.forward 内部会计算 prefix = 1 + n_storage
            # 然后执行 q = rope_apply(q[:, :, prefix:, :], sin, cos)
            # 这意味着传进去的 rope 长度必须【严格等于】输入 Patch 的长度
            # 因此这里不需要加 cls_rope 的全 0 占位符
            
            # c. 插入 Heads 维度对齐广播: [B, L_keep, 64] -> [B, 1, L_keep, 64]
            r_final = r_selected.unsqueeze(1) 
            
            rope_final_list.append(r_final)
        
        rope_final = tuple(rope_final_list)

        # 5. Blocks 前向
        for blk in self.encoder.blocks:
            # 输入 tokens 长度为 (1 + n_storage + L_keep)
            # 输入 rope 长度为 (L_keep)
            # blk 内部会自动将两者匹配
            tokens = blk(tokens, rope_final)

        # 最后进行标准化
        tokens = self.encoder.norm(tokens)
        
        return tokens, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :] # remove cls token

    def forward_text(self, text, return_all_tokens=False):
        """
        text: [B, 256] token IDs
        """
        if return_all_tokens:
            # 直接调用内部的 HF BERT transformer
            # x.last_hidden_state: [B, L, 768]
            res = self.text_encoder.transformer(text, attention_mask=(text > 0).long())
            return res.last_hidden_state
        
        # [B, 512]
        return self.text_encoder(text)

    def forward_contrastive_loss(self, visual_features, text_features_projected):
        """
        Args:
            visual_features: [B, D_vis] (Aggregated locally)
            text_features_projected: [B, D_proj]   (Processed locally)
        """
        # 1. Local Projection & Normalization
        image_embeds = F.normalize(self.image_projection(visual_features), p=2, dim=-1)
        text_embeds = F.normalize(text_features_projected, p=2, dim=-1)
        
        # 2. Distributed All-Gather (Global alignment)
        if dist.is_initialized():
            # Gather all embeddings from all GPUs
            # global_image_embeds shape: [B * World_Size, D_joint]
            global_image_embeds = all_gather_features(image_embeds.detach())
            global_text_embeds = all_gather_features(text_embeds.detach())
            
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            batch_size = image_embeds.shape[0]
            
            # 3. Calculate Similarity Matrix
            t = self.logit_scale.float().exp()
            
            # Local images vs Global texts [B, B * World_Size]
            logits_per_image = t * image_embeds @ global_text_embeds.t()
            # Local texts vs Global images
            logits_per_text = t * text_embeds @ global_image_embeds.t()
            
            # 4. Generate Global Labels
            # The correct match for the i-th local sample is at index (rank * batch_size + i)
            labels = torch.arange(batch_size, device=image_embeds.device) + rank * batch_size
            
        else:
            # Fallback to local contrastive if not distributed
            t = self.logit_scale.exp()
            logits_per_image = t * image_embeds @ text_embeds.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(image_embeds.shape[0], device=image_embeds.device)

        # 5. Cross Entropy Loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2

    def forward_fg_contrastive_loss(self, visual_features, text_features_projected, text_normal_flags, text_anatomy_flags, image_counts, sub_text_counts, K=7):
        """
        Args:
            visual_features: [Total_Images, D_vis]
            text_features_projected: [Total_sub_texts, D_txt]
            visual_lengths: List or Tensor of images per sample [n1, n2, ...]
            sub_text_lengths: List or Tensor of sub_texts per sample [m1, m2, ...]
        """
        #  Local Projection & Normalization
        image_embeds = F.normalize(self.image_projection_fg(visual_features), p=2, dim=-1, eps=1e-6) # [Total_Images, D_joint]
        text_embeds = F.normalize(text_features_projected, p=2, dim=-1, eps=1e-6)

        #  Padding and Stack global_text_embeds
        split_embeds = torch.split(text_embeds, sub_text_counts, dim=0)
        text_embeds_padding_stack = pad_sequence(split_embeds, batch_first=True) # [B, Max_sub_text, D_joint]
        
        #  Find Top-k Matched Images for each sentence in each case
        t = self.logit_scale_fg.exp()
        sent2img_similarity_raw = t * text_embeds_padding_stack @ image_embeds.transpose(-2, -1) # [B, Max_sub_text, Total_Images]
  
        #  Distributed All-Gather (Global alignment) 
        # global_image_embeds = all_gather_with_variable_size(image_embeds) 
        # global_text_embeds = all_gather_with_variable_size(text_embeds) 
        
        paired_img_embeds, all_selected_img_idxs = get_paired_img_embeds(sent2img_similarity_raw, image_embeds, image_counts, sub_text_counts, K=K)
        active_vis_feat, active_txt_feat = flatten_active_features(paired_img_embeds, text_embeds, sub_text_counts)
        active_txt_normal_flags = text_normal_flags
        active_txt_anatomy_flags = text_anatomy_flags

        if not dist.is_initialized():
            # 非分布式逻辑
            t = self.logit_scale.exp()
            logits_per_viz = t * active_vis_feat @ active_txt_feat.t()
            logits_per_txt = logits_per_viz.t()
            labels = torch.arange(active_vis_feat.shape[0], device=active_vis_feat.device)
            loss = (F.cross_entropy(logits_per_viz, labels) + F.cross_entropy(logits_per_txt, labels)) / 2
            return loss, all_selected_img_idxs

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = active_vis_feat.device

        # --- 1. 收集全量特征 ---
        # global_*_feat: [Total_Sub_global, D]
        global_vis_feat = all_gather_with_variable_size(active_vis_feat.detach())
        global_txt_feat = all_gather_with_variable_size(active_txt_feat.detach())
        global_txt_normal_flags = all_gather_with_variable_size(active_txt_normal_flags.detach())
        global_txt_anatomy_flags = all_gather_with_variable_size(active_txt_anatomy_flags.detach())
        # print(global_vis_feat.shape, global_txt_feat.shape, global_txt_normalize_flags.shape)

        # --- 2. 计算当前卡在全局矩阵中的标签偏移量 ---
        # 我们需要知道其它卡贡献了多少个子句
        local_n = torch.tensor([active_vis_feat.shape[0]], device=device)
        all_ns = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_ns, local_n)
        
        # 转换成 list [n0, n1, n2...]
        all_ns = [n.item() for n in all_ns]
        # 计算当前 rank 之前的子句总数作为 offset
        offset = sum(all_ns[:rank])
        
        # --- 3. 计算 Logits ---
        t = self.logit_scale_fg.float().exp()
        
        # 本地视觉子句 vs 全局文本子句 [Total_Sub_local, Total_Sub_global]
        logits_per_viz = t * active_vis_feat @ global_txt_feat.t()
        # 本地文本子句 vs 全局视觉子句 [Total_Sub_local, Total_Sub_global]
        logits_per_txt = t * active_txt_feat @ global_vis_feat.t()

        # --- 4. 生成 Labels ---
        labels = get_anatomical_corrected_labels(active_txt_feat, global_txt_feat, active_txt_normal_flags, global_txt_normal_flags, active_txt_anatomy_flags, global_txt_anatomy_flags) # [Total_Sub_local, Total_Sub_global]
        # labels = torch.arange(active_vis_feat.shape[0], device=device) + offset # 每一个本地子句 i 在全局列表中的索引是 offset + i

        # --- 5. 计算 Loss ---
        # loss_v2t = F.cross_entropy(logits_per_viz, labels)
        # loss_t2v = F.cross_entropy(logits_per_txt, labels)

        loss_v2t = -torch.sum(labels * F.log_softmax(logits_per_viz, dim=-1), dim=-1).mean()
        loss_t2v = -torch.sum(labels * F.log_softmax(logits_per_txt, dim=-1), dim=-1).mean()

        return (loss_v2t + loss_t2v) / 2, all_selected_img_idxs

    def forward_reconstruction_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()
        return loss

    def packed_mean_pooling(self, x_packed, lengths):
        """
        x_packed: [Sum(T), L] - all packed (flattened) instances
        lengths: [B] - read lengths of each bag
        return: [B, L] - mean pooled features
        """
        device = x_packed.device
        L = x_packed.shape[-1]
        B = len(lengths)
        
        # generate Batch Index [0,0,0, 1,1, 2,2,2,2...]
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=device), 
            torch.tensor(lengths, device=device)
        )
        
        # sum_features: [B, L]
        sum_features = torch.zeros(B, L, device=device, dtype=x_packed.dtype)
        
        # Scatter Add
        sum_features.index_add_(0, batch_idx, x_packed)
        
        count = torch.tensor(lengths, device=device).float().unsqueeze(-1).clamp(min=1e-9)
        mean_features = sum_features / count
        
        return mean_features

    def forward(self, imgs, full_texts, sub_texts, text_normal_flags, text_anatomy_flags, image_counts, sub_text_counts, mask_ratio=0.75, K=7, mode='both'):
        """
        Args:
            imgs: [Total_Images, C, H, W] - All images flattened into one batch
            full_texts: [Batch_Size, L] - One text per sample
            sub_texts: [Total_sub_texts, L] - All sub-texts flattened into one batch
            text_normal_flags: [Total_sub_texts] - One flag per sub-text
            text_anatomy_flags: [Total_sub_texts] - One flag per sub-text
            image_counts: List or Tensor of images per sample [n1, n2, ...]
            sub_text_counts:  List or Tensor of sub_texts per sample [n1, n2, ...]
            mask_ratio: float - Percentage of patches to mask
            mode: 'both', 'mae_only', 'clip_only'
        """
        # 1. Shared Encoder (treats all images as independent in the batch dimension)
        latent, _, _ = self.forward_encoder(imgs, 0.0)
        
        loss_mae = torch.tensor(0.0, device=imgs.device)
        loss_clip = torch.tensor(0.0, device=imgs.device)
        
        pred, unique_selected_patch_idxs = None, None

        # 2. CLIP Branch (Per-sample alignment)
        if mode in ['both', 'clip_only', 'ablation']:
            # Extract Per-image visual features: [Total_Images, D]
            cls_tokens = latent[:, 0] # [Total_Images, D]
            patch_tokens = latent[:, 1 + self.encoder.n_storage_tokens:] # [Total_Images, L, D]
            per_visual_features = self.image_feat_projection(torch.cat([cls_tokens, patch_tokens.mean(dim=1)], dim=1))
            # Aggregate to per-sample visual features: [Batch_Size, D]
            global_visual_features = self.packed_mean_pooling(per_visual_features, image_counts)
            
            # Text features: [Batch_Size, D_txt]
            global_text_features = self.forward_text(full_texts)
            # Sub-text features: [Total_Sub, D_txt]
            sub_text_features = self.forward_text(sub_texts)
            
            # Align
            # print(per_visual_features.shape, sub_text_features.shape)
            # print(global_visual_cls.shape, global_text_features.shape)

            
            # loss_clip_global = self.forward_contrastive_loss(global_visual_features, global_text_features)
            # loss_clip_FG, all_selected_indices = self.forward_fg_contrastive_loss(per_visual_features, sub_text_features, text_normal_flags, text_anatomy_flags, image_counts, sub_text_counts, K=K)
            # loss_clip = loss_clip_global + loss_clip_FG
            
            # Ablation -- only L_fg
            loss_clip_global = torch.tensor(0.0, device=imgs.device)
            loss_clip_FG, all_selected_indices = self.forward_fg_contrastive_loss(per_visual_features, sub_text_features, text_normal_flags, text_anatomy_flags, image_counts, sub_text_counts, K=K)
            loss_clip = loss_clip_FG

        # 3. Semantic Focus MAE Branch (Per-image reconstruction)
        if mode in ['both', 'mae_only']:
            # --- 整理全局索引列表 ---
            if len(all_selected_indices) > 0:
                total_selected_tensor = torch.cat(all_selected_indices)
                unique_selected_patch_idxs = torch.unique(total_selected_tensor)
            else:
                unique_selected_patch_idxs = torch.arange(0, len(imgs), dtype=torch.long, device=device)
            imgs_mae = imgs[unique_selected_patch_idxs]
            latent_mae, mask_mae, ids_restore_mae = self.forward_encoder(imgs_mae, mask_ratio)
            pred = self.forward_decoder(latent_mae, ids_restore_mae)
            loss_mae = self.forward_reconstruction_loss(imgs_mae, pred, mask_mae)

        return loss_mae + loss_clip, loss_mae, loss_clip_global, loss_clip_FG, unique_selected_patch_idxs

@torch.no_grad()
def get_corrected_labels(local_text_embeds, global_text_embeds, local_normal_flags, global_normal_flags, local_anatomy_flags, global_anatomy_flags):
    """
    Args:
        local_text_embeds: [N_local, D] - 本地卡的子句文本特征 (已归一化)
        global_text_embeds: [N_global, D] - 所有卡的子句文本特征 (已归一化)
        local_normal_flags: [N_local] - 本地子句是否为“正常”的布尔值
        global_normal_flags: [N_global] - 全域子句是否为“正常”的布尔值
        local_anatomy_flags: [N_local] - 本地子句为哪一个解剖部位的值，[0, 1, 2, ..., 16]
    Returns:
        soft_labels: [N_local, N_global]
    """
    device = local_text_embeds.device
    num_local = local_text_embeds.shape[0]
    num_global = global_text_embeds.shape[0]

    # 1. 计算文本语义相似度 (用于异常情况)
    # text_sim 范围通常在 [-1, 1], 我们将其限制在 [0, 1] 以符合概率逻辑
    text_sim = torch.matmul(local_text_embeds, global_text_embeds.t())
    text_sim = torch.clamp(text_sim, min=0.0)

    # 2. 构造双正常掩码 (Both Normal Mask)
    # [N_local, 1] & [1, N_global] -> [N_local, N_global]
    # 只有当 i 和 j 都是正常描述时，该位置为 True
    # both_normal_mask = local_normal_flags.unsqueeze(1) & global_normal_flags.unsqueeze(0)
    both_normal_mask = torch.outer(local_normal_flags.bool(), global_normal_flags.bool())

    # 3. 构造初始 Targets
    # 如果两个都是正常，Target 设为 1.0
    # 否则，使用文本相似度作为 Soft Target
    soft_labels = torch.where(both_normal_mask, torch.ones_like(text_sim), text_sim)

    # 4. 强制对角线为 1 (确保 local 样本始终和自己的文本对齐)
    # 首先需要找到当前卡在全局索引中的起始位置 (offset)
    if dist.is_initialized():
        rank = dist.get_rank()
        sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, torch.tensor([num_local], device=device))
        offset = sum([s.item() for s in sizes[:rank]])
    else:
        offset = 0
    
    # 填充对角线
    diag_indices = torch.arange(num_local, device=device)
    soft_labels[diag_indices, diag_indices + offset] = 1.0

    # 5. 归一化
    # 每一行的和代表了该图像对应的“所有合法的文本解释”的权重
    soft_labels = soft_labels / torch.clamp(soft_labels.sum(dim=1, keepdim=True), min=1e-6)

    return soft_labels

@torch.no_grad()
def get_anatomical_corrected_labels(
    local_text_embeds, global_text_embeds, 
    local_normal_flags, global_normal_flags, 
    local_anatomy_flags, global_anatomy_flags
):
    """
    Args:
        local_text_embeds: [N_local, D]
        global_text_embeds: [N_global, D]
        local_normal_flags: [N_local] (Bool or 0/1)
        global_normal_flags: [N_global] (Bool or 0/1)
        local_anatomy_flags: [N_local] (int, e.g., 0-16)
        global_anatomy_flags: [N_global] (int, e.g., 0-16)
    Returns:
        soft_labels: [N_local, N_global]
    """
    device = local_text_embeds.device
    num_local = local_text_embeds.shape[0]
    num_global = global_text_embeds.shape[0]

    # --- 1. 基础语义相似度 (Base Semantic Similarity) ---
    # Case 2 & 3 的基础：利用文本编码器的能力
    text_sim = torch.matmul(local_text_embeds, global_text_embeds.t())
    text_sim = torch.clamp(text_sim, min=0.0, max=1.0) # 限制在 [0, 1]

    # --- 2. 构建掩码 (Mask Construction) ---
    
    # A. 解剖一致性掩码 (Anatomical Consistency Mask)
    # [N_local, 1] == [1, N_global] -> [N_local, N_global]
    # 只有部位相同时为 True
    anatomy_match_mask = (local_anatomy_flags.unsqueeze(1) == global_anatomy_flags.unsqueeze(0))

    # B. 双正常掩码 (Both Normal Mask)
    # 只有两者都为正常时为 True
    both_normal_mask = torch.outer(local_normal_flags.bool(), global_normal_flags.bool())

    # --- 3. 标签修正逻辑 (Label Correction Logic) ---
    
    # 初始化 soft_labels 为 text_sim (涵盖了 Case 2: 同部位异常 & Case 3: 同部位一正一异)
    soft_labels = text_sim

    # 逻辑层级 1 (Case 1): 同部位 + 双正常 -> 强制置为 1.0
    # 注意：这里隐式包含了 anatomy matching，因为不同部位的双正常不应为1，
    # 但我们需要结合 anatomy_match_mask 来看。
    # 修正逻辑：只有在 (部位相同 AND 双正常) 时，才置为 1.0
    # 实际上，只要部位相同，且双正常，就是1。如果部位不同，即便双正常也是0 (由层级2处理)。
    
    # 我们先应用双正常逻辑 (假设部位相同)
    soft_labels = torch.where(both_normal_mask, torch.ones_like(soft_labels), soft_labels)

    # 逻辑层级 2 (Case 4): 部位不同 -> 强制置为 0.0 (Hard Negative)
    # 这是最高优先级的过滤器
    soft_labels = soft_labels * anatomy_match_mask.float()

    # --- 4. 对角线一致性 (Self-Correction) ---
    # 确保样本自己和自己永远是对齐的 (即使在上述逻辑中有误判，自身对齐是Ground Truth)
    if dist.is_initialized():
        rank = dist.get_rank()
        sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, torch.tensor([num_local], device=device))
        offset = sum([s.item() for s in sizes[:rank]])
    else:
        offset = 0
    
    diag_indices = torch.arange(num_local, device=device)
    # 只有当全局索引在当前batch范围内时才设置 (防止越界)
    if offset + num_local <= num_global:
        soft_labels[diag_indices, diag_indices + offset] = 1.0
    
    # --- 5. 概率归一化 (Probability Normalization) ---
    # 让每一行成为一个合法的概率分布
    row_sum = soft_labels.sum(dim=1, keepdim=True)
    soft_labels = soft_labels / torch.clamp(row_sum, min=1e-6)

    return soft_labels


def get_paired_img_embeds(sent2img_similarity_raw, image_embeds, image_counts, sub_text_counts, K=7):
    """
    Args:
        sent2img_similarity_raw: [B, Max_sub, Total_Imgs]
        image_embeds: [Total_Imgs, D]
        image_counts: List[int]
        sub_text_counts: List[int]
    Returns:
        paired_img_embeds: [B, Max_sub, D]
        unique_selected_patch_idxs: torch.Tensor [Num_Unique_Selected] - 全局索引列表
    """
    B, M, _ = sent2img_similarity_raw.shape
    D = image_embeds.shape[-1]
    device = image_embeds.device
    
    paired_img_embeds = torch.zeros(B, M, D, device=device, dtype=image_embeds.dtype)
    img_offsets = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            torch.cumsum(torch.tensor(image_counts, device=device, dtype=torch.long), dim=0)
        ])


    # 用于收集被选中的全局索引
    all_selected_indices = []

    for i in range(B):
        n_i = image_counts[i]
        s_i = sub_text_counts[i]
        if s_i == 0 or n_i == 0:
            continue
            
        start, end = img_offsets[i], img_offsets[i+1]
        
        # 1. 提取局部相似度 [s_i, n_i]
        local_sim = sent2img_similarity_raw[i, :s_i, start:end]
        
        # 2. 取 Top-K
        actual_k = min(K, n_i)
        topk_val, topk_idx = local_sim.topk(actual_k, dim=-1) # [s_i, k]
        # print(start, topk_idx)
        
        # --- 新增逻辑：计算并记录全局索引 ---
        # topk_idx 目前是相对于当前样本图片池 [0, n_i-1] 的偏移
        # 加上 start 偏移量，变为相对于 Total_Imgs [0, Total_Imgs-1] 的全局索引
        global_topk_idx = topk_idx + start
        # print(start, global_topk_idx)
        all_selected_indices.append(global_topk_idx.reshape(-1))
        # ----------------------------------

        # 3. 归一化权重
        weights = F.softmax(topk_val / 0.07, dim=-1) 
        
        # 4. 提取特征
        local_img_embeds = image_embeds[start:end]
        selected_features = local_img_embeds[topk_idx]
        
        # 5. 加权聚合
        paired_feat = torch.bmm(weights.unsqueeze(1), selected_features).squeeze(1)
        paired_img_embeds[i, :s_i] = paired_feat
        
    return paired_img_embeds, all_selected_indices


def flatten_active_features(paired_img_embeds, sub_text_features, sub_text_counts):
    """
    Args:
        paired_img_embeds: [B, Max_sub, D] - 已经通过 Top-K 聚合好的视觉特征
        sub_text_features: [Total_Sub_local, D] - 文本端直接输出的子句特征
        sub_text_counts: List[int] [s1, s2, ..., sB]
    """
    B, M, D = paired_img_embeds.shape
    device = paired_img_embeds.device
    
    # 创建掩码提取非 Padding 部分
    # mask: [B, M]
    mask = torch.arange(M, device=device).unsqueeze(0) < torch.tensor(sub_text_counts, device=device).unsqueeze(1)
    
    # 提取有效视觉特征: [Total_Sub_local, D]
    active_vis_feat = paired_img_embeds[mask] 
    
    # 归一化
    active_vis_feat = F.normalize(active_vis_feat, p=2, dim=-1, eps=1e-6)
    active_txt_feat = F.normalize(sub_text_features, p=2, dim=-1, eps=1e-6)
    
    return active_vis_feat, active_txt_feat

# --- Debug / Testing Module ---
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Endoscopic images often vary: Batch of 3 samples, with 2, 1, and 3 images respectively
    img_counts = [12, 11, 13]
    total_imgs_count = sum(img_counts)

    text_counts = [8,8,9]
    total_texts_count = sum(text_counts)
    
    # model = MaskedAutoencoder_MuImg_ViT(
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    # ).to(device)

    # # Inputs
    # dummy_imgs = torch.randn(total_imgs_count, 3, 224, 224).to(device) 
    # dummy_total_text = torch.randint(0, 49408, (len(text_counts), 77)).to(device) # One text per sample
    # dummy_sub_text = torch.randint(0, 49408, (total_texts_count, 77)).to(device)
    # text_normal_flags = torch.randint(0, 2, (total_texts_count,)).to(device)
    # text_anatomy_flags = torch.randint(0, 17, (total_texts_count,)).to(device)

    # # Forward
    # loss, loss_mae, loss_clip_global, loss_clip_FG, unique_selected_patch_idxs = model(dummy_imgs, dummy_total_text, dummy_sub_text, text_normal_flags, text_anatomy_flags, image_counts=img_counts, sub_text_counts=text_counts, mode='both', K=3)

    # print(f"Total Images: {total_imgs_count}, Samples: {len(img_counts)}")
    # print(f"Loss -> MAE: {loss_mae.item():.4f}, global_CLIP: {loss_clip_global.item():.4f}, FG_CLIP: {loss_clip_FG.item():.4f}")
    # print(f"Unique Selected Patch Idxs: {unique_selected_patch_idxs}") # length < K * total_texts_count


