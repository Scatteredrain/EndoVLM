import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
import open_clip
from functools import partial
import torch.distributed as dist

def all_gather_features(features):
    """
    Collect features from all GPU nodes with gradient support.
    """
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


class MaskedAutoencoder_MuImg_ViT(nn.Module):
    '''
    Masked Autoencoder for Multi-Image -- Text Interaction
    '''
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mode='mae', vit_pretrain='mae'):
        super().__init__()

        # --- MAE Encoder ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --- MAE Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        # --- CLIP Text Encoder ---
        # Note: We load weights temporarily to extract parameters
        model_open_clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        if mode in ['clip', 'both']:
            self.text_transformer = model_open_clip.transformer
            self.token_embedding = model_open_clip.token_embedding
            self.text_positional_embedding = model_open_clip.positional_embedding
            self.ln_final = model_open_clip.ln_final
            self.text_projection = nn.Parameter(model_open_clip.text_projection.clone(), requires_grad=False)
            self.logit_scale = nn.Parameter(model_open_clip.logit_scale.clone())

            # Freeze text branch by default
            # self.text_transformer.requires_grad_(False)

            # --- Image Projection for cls_token & path_tokens aggregation ---
            self.image_feat_projection = nn.Linear(embed_dim*2, embed_dim)

            # --- Image Projection for CLIP ---
            self.image_projection = nn.Linear(embed_dim, self.text_projection.shape[1], bias=False)
        
        # --- Initialization Flow ---
        # 1. First, generic random initialization
        self.initialize_weights()
        # 2. Second, override visual encoder and projection with pretrained weights
        if vit_pretrain == 'clip':
            self.load_clip_visual_weights(model_open_clip, mode)
        elif vit_pretrain == 'mae':
            self.load_mae_pretrain('/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/pretrained/mae_pretrain_vit_base.pth')
        self.norm_pix_loss = norm_pix_loss

    def initialize_weights(self):
        # Sine-cos pos_embed initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Basic init
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_mae_pretrain(self, checkpoint_path):
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
            if k in self.state_dict():
                # 检查形状是否匹配 (防止 Base 加载到 Large 上)
                if v.shape == self.state_dict()[k].shape:
                    new_state_dict[k] = v
                else:
                    print(f"Shape mismatch for {k}: skipping...")

        # 2. 执行加载
        # 使用 strict=False，因为我们故意漏掉了 Decoder 的权重
        msg = self.load_state_dict(new_state_dict, strict=False)
        
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

    def load_clip_visual_weights(self, model_open_clip, mode):
        clip_sd = model_open_clip.visual.state_dict()
        new_sd = self.state_dict()
        print("Mapping CLIP weights...")
        
        # Mapping logic (Simplified for display, keep your existing mapping logic here)
        new_sd["patch_embed.proj.weight"].copy_(clip_sd["conv1.weight"])
        new_sd["cls_token"].copy_(clip_sd["class_embedding"].reshape(1, 1, -1))
        new_sd["pos_embed"].copy_(clip_sd["positional_embedding"].unsqueeze(0))
        
        for i in range(len(self.blocks)):
            new_sd[f"blocks.{i}.attn.qkv.weight"].copy_(clip_sd[f"transformer.resblocks.{i}.attn.in_proj_weight"])
            new_sd[f"blocks.{i}.attn.qkv.bias"].copy_(clip_sd[f"transformer.resblocks.{i}.attn.in_proj_bias"])
            new_sd[f"blocks.{i}.attn.proj.weight"].copy_(clip_sd[f"transformer.resblocks.{i}.attn.out_proj.weight"])
            new_sd[f"blocks.{i}.attn.proj.bias"].copy_(clip_sd[f"transformer.resblocks.{i}.attn.out_proj.bias"])
            # ... add other blocks mapping (norm1, norm2, mlp.fc1, mlp.fc2)
            
        new_sd["norm.weight"].copy_(clip_sd["ln_post.weight"])
        new_sd["norm.bias"].copy_(clip_sd["ln_post.bias"])
        
        if "proj" in clip_sd and mode in ['clip', 'both']:
            new_sd["image_projection.weight"].copy_(clip_sd["proj"].t())
        
        self.load_state_dict(new_sd)
        print("CLIP weights loaded.")

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x) # [N, L, D]
        x = x + self.pos_embed[:, 1:, :] 
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1) # [N, L+1, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

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

    def forward_text(self, text):
        x = self.token_embedding(text)
        x = x + self.text_positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return x

    # def forward_contrastive_loss(self, visual_features, text_features):
    #     image_embeds = F.normalize(self.image_projection(visual_features), p=2, dim=-1)
    #     text_embeds = F.normalize(text_features @ self.text_projection, p=2, dim=-1)
        
    #     t = self.logit_scale.exp()
    #     logits_per_image = t * image_embeds @ text_embeds.t()
    #     logits_per_text = logits_per_image.t()
        
    #     labels = torch.arange(image_embeds.shape[0], device=image_embeds.device)
    #     loss_clip = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
    #     return loss_clip

    def forward_contrastive_loss(self, visual_features, text_features):
        """
        Args:
            visual_features: [B, D_vis] (Aggregated locally)
            text_features: [B, D_txt]   (Processed locally)
        """
        # 1. Local Projection & Normalization
        image_embeds = F.normalize(self.image_projection(visual_features), p=2, dim=-1)
        text_embeds = F.normalize(text_features @ self.text_projection, p=2, dim=-1)
        
        # 2. Distributed All-Gather (Global alignment)
        if dist.is_initialized():
            # Gather all embeddings from all GPUs
            # global_image_embeds shape: [B * World_Size, D_joint]
            global_image_embeds = all_gather_features(image_embeds)
            global_text_embeds = all_gather_features(text_embeds)
            
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            batch_size = image_embeds.shape[0]
            
            # 3. Calculate Similarity Matrix
            t = self.logit_scale.exp()
            
            # Local images vs Global texts
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

    def forward(self, imgs, text, image_counts, mask_ratio=0.75, mode='both'):
        """
        Args:
            imgs: [Total_Images, C, H, W] - All images flattened into one batch
            text: [Batch_Size, L] - One text per sample
            image_counts: List or Tensor of images per sample [n1, n2, ...]
            mode: 'both', 'mae_only', 'clip_only'
        """
        # 1. Shared Encoder (treats all images as independent in the batch dimension)
        # print(imgs.shape)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        loss_mae = torch.tensor(0.0, device=imgs.device)
        loss_clip = torch.tensor(0.0, device=imgs.device)
        pred = None

        # 2. MAE Branch (Per-image reconstruction)
        if mode in ['both', 'mae']:
            pred = self.forward_decoder(latent, ids_restore)
            loss_mae = self.forward_reconstruction_loss(imgs, pred, mask)
            
        # 3. CLIP Branch (Per-sample alignment)
        if mode in ['both', 'clip']:
            # Extract per-image CLS tokens: [Total_Images, D]
            # per_image_cls = latent[:, 0]
            cls_tokens = latent[:, 0] # [Total_Images, D]
            patch_tokens = latent[:, 1:] # [Total_Images, L, D]
            per_visual_features = self.image_feat_projection(torch.cat([cls_tokens, patch_tokens.mean(dim=1)], dim=1))
            
            # Aggregate to per-sample visual features: [Batch_Size, D]
            sample_visual_features = self.packed_mean_pooling(per_visual_features, image_counts)
            
            # Text features: [Batch_Size, D_txt]
            text_features = self.forward_text(text)
            
            # Align
            loss_clip = self.forward_contrastive_loss(sample_visual_features, text_features)

        return loss_mae + loss_clip, loss_mae, loss_clip


# --- Debug / Testing Module ---
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Endoscopic images often vary: Batch of 3 samples, with 2, 1, and 3 images respectively
    img_counts = [2, 1, 3]
    total_imgs_count = sum(img_counts)
    
    model = MaskedAutoencoder_MuImg_ViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    ).to(device)

    # Inputs
    dummy_imgs = torch.randn(total_imgs_count, 3, 224, 224).to(device) 
    dummy_text = torch.randint(0, 49408, (len(img_counts), 77)).to(device) # One text per sample

    # Forward
    loss, mae, clip = model(dummy_imgs, dummy_text, image_counts=img_counts, mode='both')
    
    print(f"Total Images: {total_imgs_count}, Samples: {len(img_counts)}")
    print(f"Loss -> MAE: {mae.item():.4f}, CLIP: {clip.item():.4f}")