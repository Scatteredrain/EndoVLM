# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import timm.models.vision_transformer
import open_clip
import timm
import sys
import traceback

CONFIG = {
    # mae 官方预训练权重路径
    "mae_weight_path": "/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/pretrained/mae_pretrain_vit_base.pth",
    # clip_ours 自定义权重路径（需要包含 'model' 键的 checkpoint）
    "clip_ours_weight_path": "/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_clip/2026-02-08-03_31_28/checkpoint-6.pth",
    # mae_ours 自定义权重路径（需要包含 'model' 键的 checkpoint）
    "mae_ours_weight_path": "/mnt/data/yizhenyu/data/Endomaster/workspace/VLP/EndoVLP/output_dir_mae/2026-02-04-07_02_43/checkpoint-10.pth",
}


def load_weights(model_raw, mode='clip', weight_path=None):
    if mode == 'clip':
        # model_open_clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        model_open_clip = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True)
        checkpoint_model = model_open_clip.state_dict()
    elif mode == 'mae':
        weight_path = CONFIG["mae_weight_path"]
        checkpoint_model = torch.load(weight_path, map_location='cpu', weights_only=False)['model']
        # if hasattr(model_raw, 'fc_norm'):
        #     if 'norm.weight' in checkpoint_model and 'fc_norm.weight' not in checkpoint_model:
        #         checkpoint_model['fc_norm.weight'] = checkpoint_model.pop('norm.weight')
        #         checkpoint_model['fc_norm.bias'] = checkpoint_model.pop('norm.bias')
        #         print("Renamed: norm.weight/bias → fc_norm.weight/bias")
    elif mode == 'biomeclip':
        biomedclip, _ = open_clip.create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir=os.path.expanduser('~/.cache/huggingface/hub')
        )
        checkpoint_model = biomedclip.visual.trunk.state_dict()
        # print(checkpoint_model.keys())
        # print(model_raw.state_dict().keys())
    elif mode == 'clip_ours':
        weight_path = CONFIG["clip_ours_weight_path"]
        checkpoint_model = torch.load(weight_path, map_location='cpu', weights_only=False)['model']
    elif mode == 'mae_ours':
        weight_path = CONFIG["mae_ours_weight_path"]
        checkpoint_model = torch.load(weight_path, map_location='cpu', weights_only=False)['model']
    
    interpolate_pos_embed(model_raw, checkpoint_model)
    msg = model_raw.load_state_dict(checkpoint_model, strict=False)
    print(msg)

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# model = models_vit.__dict__[args.model](
#     num_classes=args.nb_classes,
#     drop_path_rate=args.drop_path,
#     global_pool=args.global_pool,
# )


############################################################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证 vit_base_patch16 在五种权重加载模式下是否正常工作
模式: clip, biomeclip, mae, clip_ours, mae_ours
"""

# 测试用的虚拟输入图像 
DUMMY_INPUT_SHAPE = (2, 3, 352, 352)


def test_mode(mode: str, weight_path: str = None, strict: bool = True):
    """
    测试单个模式的权重加载和前向推理。

    Args:
        mode: 权重加载模式
        weight_path: 权重文件路径（仅 clip_ours / mae_ours 需要）
        strict: 是否严格要求权重完全匹配
    """
    # from models_vit import vit_base_patch16, load_weights

    print("=" * 70)
    print(f"[TEST] 模式: {mode}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 1: 创建模型
    # ----------------------------------------------------------
    print(f"  [1/4] 创建 vit_base_patch16 模型...")


    model = vit_base_patch16(num_classes=0, global_pool=False, img_size=352)
    # print(model.state_dict().keys())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       模型参数量: {total_params / 1e6:.2f}M")

    # ----------------------------------------------------------
    # Step 2: 加载权重
    # ----------------------------------------------------------
    print(f"  [2/4] 加载权重 (mode='{mode}')...")

    if mode in ["clip_ours", "mae_ours"]:
        if not os.path.exists(weight_path):
            print(f"       ⚠️  权重文件不存在: {weight_path}")
            print(f"       ⚠️  跳过此模式的测试（文件缺失）")
            print()
            return False

    if mode == "mae" and not os.path.exists(CONFIG["mae_weight_path"]):
        print(f"       ⚠️  MAE 权重文件不存在: {CONFIG['mae_weight_path']}")
        print(f"       ⚠️  跳过此模式的测试（文件缺失）")
        print()
        return False

    try:
        load_weights(model, mode=mode)
        print(f"       ✅ 权重加载成功!")
    except Exception as e:
        print(f"       ❌ 权重加载失败!")
        print(f"       错误信息: {e}")
        traceback.print_exc()
        print()
        return False

    # ----------------------------------------------------------
    # Step 3: 前向推理测试
    # ----------------------------------------------------------
    print(f"  [3/4] 前向推理测试 (input shape: {DUMMY_INPUT_SHAPE})...")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dummy_input = torch.randn(DUMMY_INPUT_SHAPE).to(device)

    try:
        with torch.no_grad():
            output = model.forward_features(dummy_input)
        print(f"       ✅ 前向推理成功!")
        print(f"       输出 shape: {output.shape}")
        print(f"       输出 dtype: {output.dtype}")
        print(f"       输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"       ❌ 前向推理失败!")
        print(f"       错误信息: {e}")
        traceback.print_exc()
        print()
        return False

    # ----------------------------------------------------------
    # Step 4: 梯度计算测试
    # ----------------------------------------------------------
    print(f"  [4/4] 梯度计算测试...")

    model.train()
    dummy_input_grad = torch.randn(DUMMY_INPUT_SHAPE).to(device)

    try:
        output = model.forward_features(dummy_input_grad)
        loss = output.sum()
        loss.backward()
        
        # 检查是否有梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        if has_grad:
            print(f"       ✅ 梯度计算成功!")
        else:
            print(f"       ⚠️  梯度全部为零（可能是正常的）")
    except Exception as e:
        print(f"       ❌ 梯度计算失败!")
        print(f"       错误信息: {e}")
        traceback.print_exc()
        print()
        return False

    # 清理显存
    del model, dummy_input, dummy_input_grad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  🎉 模式 '{mode}' 全部测试通过!\n")
    return True


def main():
    print("\n" + "#" * 70)
    print("#  VIT_BASE_PATCH16 五种权重加载模式测试")
    print("#" * 70 + "\n")

    # 设备信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"测试设备: {device}")
    # if torch.cuda.is_available():
    #     print(f"GPU: {torch.cuda.get_device_name(0)}")
    #     print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    # print()

    # ----------------------------------------------------------
    # 定义五种测试模式
    # ----------------------------------------------------------
    test_cases = [
        {
            "mode": "clip",
            "weight_path": None,
            "description": "OpenAI CLIP ViT-B/16 预训练权重",
        },
        {
            "mode": "biomeclip",
            "weight_path": None,
            "description": "Microsoft BiomedCLIP 预训练权重",
        },
        {
            "mode": "mae",
            "weight_path": CONFIG["mae_weight_path"],
            "description": "MAE 官方预训练权重",
        },
        {
            "mode": "clip_ours",
            "weight_path": CONFIG["clip_ours_weight_path"],
            "description": "自定义 CLIP 微调权重",
        },
        {
            "mode": "mae_ours",
            "weight_path": CONFIG["mae_ours_weight_path"],
            "description": "自定义 MAE 微调权重",
        },
    ]

    # ----------------------------------------------------------
    # 逐一测试
    # ----------------------------------------------------------
    results = {}
    for tc in test_cases:
        mode = tc["mode"]
        print(f"📋 {tc['description']}")
        try:
            success = test_mode(
                mode=mode,
                weight_path=tc.get("weight_path"),
            )
            results[mode] = "✅ 通过" if success else "⚠️ 跳过/失败"
        except Exception as e:
            results[mode] = f"❌ 异常: {e}"
            traceback.print_exc()
            print()

    # ----------------------------------------------------------
    # 汇总结果
    # ----------------------------------------------------------
    print("\n" + "#" * 70)
    print("#  测试结果汇总")
    print("#" * 70)
    for mode, result in results.items():
        print(f"  {mode:<15s} : {result}")
    print("#" * 70 + "\n")

    # 检查是否全部通过
    all_passed = all("通过" in v for v in results.values())
    if all_passed:
        print("🎉 所有模式测试通过!")
    else:
        print("⚠️  部分模式未通过，请检查上方日志。")
        # 对于缺失文件的情况给出提示
        print("\n💡 提示：")
        print("   - clip / biomeclip: 需要网络下载预训练权重")
        print(f"   - mae:             需要文件 {CONFIG['mae_weight_path']}")
        print(f"   - clip_ours:       需要文件 {CONFIG['clip_ours_weight_path']}")
        print(f"   - mae_ours:        需要文件 {CONFIG['mae_ours_weight_path']}")


if __name__ == "__main__":
    main()
