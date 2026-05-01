# Copyright (c) RSRefSeg. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (c) 2025 DJ_W.
# Description: "The model structure has been reconstructed, integrating geographical text information and location data."

from typing import Optional, List, Tuple
import einops
from peft import LoraConfig, get_peft_model
from torchvision.transforms.functional import normalize
from modelscope import snapshot_download
from mmengine.model import BaseModule
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor, SamConfig, SamModel, SiglipModel, SamProcessor, SiglipProcessor
from mmseg.models import EncoderDecoder, accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
import os
import random
from mmengine.structures import PixelData
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from mmengine.structures import LabelData
from torchvision.models import resnet50
from transformers.modeling_outputs import BaseModelOutputWithPooling # 关键：引入标准输出类


"""
    消融实验：将siglip vision encoder替换为resnet
"""

#########################设置随机种子
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)
os.environ['PYTHONHASHSEED'] = str(3407)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(3407)     # 为CPU设置随机种子
torch.cuda.manual_seed(3407)      # 为当前GPU设置随机种子（只用一块GPU）
#########################设置随机种子


@MODELS.register_module()
class RefSegEncoderDecoder(EncoderDecoder):
    def __init__(
            self,
            lora_cfg: dict,
            freeze_cfg: dict,
            clip_vision_encoder: str,
            clip_text_encoder: str,
            sam_prompt_encoder: str,
            sam_mask_decoder: str,
            seg_head: dict,
            cls_head: dict,
            dummy_head=dict(type='FCNHead', in_channels=256, channels=16, num_classes=2), # 假的分类头，防止继承EncoderDecoder出现decode_head报错
            *args,
            norm_cfg: ConfigType = dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg: ConfigType = dict(type='GELU'),
            ctr_loss_weight: float = 0.5,  # 对比学习损失权重
            **kwargs):
        super().__init__(decode_head=dummy_head, *args, **kwargs)
        self.lora_cfg = lora_cfg # 保存 LoRA 微调配置
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.freeze_cfg = freeze_cfg
        self.ctr_loss_weight = ctr_loss_weight

        self.clip_vision_encoder = MODELS.build(clip_vision_encoder) # 构建 CLIP 视觉编码器模块
        self.clip_text_encoder = MODELS.build(clip_text_encoder) # 构建 CLIP 文本编码器模块
        self.sam_prompt_encoder = MODELS.build(sam_prompt_encoder)
        self.sam_mask_decoder = MODELS.build(sam_mask_decoder)
        self.seg_head = MODELS.build(seg_head) # 初始化 分割监督头

        # 自动获取 SigLIP 隐藏层维度
        if hasattr(self.clip_vision_encoder, 'config'):
            clip_dim = self.clip_vision_encoder.config.hidden_size
        else:
            clip_dim = 1024
        # 初始化 建筑分类头 (Main Task)
        if cls_head.get('in_channels') is None:
            cls_head['in_channels'] = clip_dim
        self.cls_head = MODELS.build(cls_head)

        # 对比学习投影头 (Projection Heads) # 在独立空间进行对齐，保护原始视觉/文本特征不被对比损失破坏
        self.v_proj_head = nn.Sequential(
            nn.Linear(clip_dim, clip_dim // 2),
            nn.GELU(),
            nn.Linear(clip_dim // 2, 256)
        )
        self.t_proj_head = nn.Sequential(
            nn.Linear(clip_dim, clip_dim // 2),
            nn.GELU(),
            nn.Linear(clip_dim // 2, 256)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 可学习的对比温度参数

        # # 将文本向量的空间投影映射到视觉维度
        # self.text_to_vis_proj = nn.Conv2d(
        #     in_channels  = clip_dim,  # SigLIP 文本 Token 维度
        #     out_channels = clip_dim,  # SigLIP 视觉 Patch 维度
        #     kernel_size  = 1
        # )

        self.simple_concat_proj = nn.Sequential(
            nn.Conv2d(clip_dim * 2, clip_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, clip_dim),
            nn.GELU()
        )

        # 用于对 SAM 的特征图进行通道注意力加权
        self.context_gate = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.Sigmoid()
        )

        # self.poi_scale = nn.Parameter(torch.zeros(1))

        self.set_finetune_parameters()    # 执行微调参数设置（冻结/解冻/LoRA注入）
        self.print_trainable_parameters() # 打印可训练参数统计信息
        self.print_model_structure()      # 打印网络结构信息


    def set_finetune_parameters(self):
        # 配置 LoRA 微调策略
        # peft_keys = ['backbone', 'clip_text_encoder', 'clip_vision_encoder']
        peft_keys = ['backbone', 'clip_text_encoder']
        for k in peft_keys:
            if hasattr(self, k):
                wrapper_module = getattr(self, k)
                if wrapper_module is None:
                    continue
                # 如果该模块在 LoRA 配置中
                if k in self.lora_cfg:
                    v_ = self.lora_cfg[k].copy()
                    lora_config = LoraConfig(**v_)
                    # 优先对内部的 model 属性注入 LoRA（针对 HuggingFace 包装类）
                    if hasattr(wrapper_module, 'model'):
                        inner_model = wrapper_module.model
                        if not hasattr(inner_model, 'peft_config'):
                            # 替换模型为 PeftModel，自动冻结非 LoRA 参数
                            wrapper_module.model = get_peft_model(inner_model, lora_config)
                            print(f"Applied LoRA to {k}.model")
                    else:
                        # 如果没有 .model，直接对模块本身注入
                        if not hasattr(wrapper_module, 'peft_config'):
                            setattr(self, k, get_peft_model(wrapper_module, lora_config))
                else:
                    # 如果不在配置中，完全冻结该模块
                    print(f"Info: {k} not in lora_cfg, freezing.")
                    wrapper_module.requires_grad_(False)

        # 冻结resnet中的结构
        # 执行冻结 # trainable params: 19,382,219 || all params: 695,041,037 || trainable%: 2.7886
        if hasattr(self, "clip_vision_encoder") and "clip_vision_encoder" in self.freeze_cfg:
            resnet_model = self.clip_vision_encoder.model
            freeze_layers = self.freeze_cfg["clip_vision_encoder"].get("freeze_layers", [])
            # 执行冻结
            for name, module in resnet_model.named_children():
                if name in freeze_layers:
                    for p in module.parameters():
                        p.requires_grad = False
                    print(f"Froze ResNet layer: {name}")


    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        trainable_params = 0
        all_param = 0
        # 遍历所有参数统计数量
        for _, param in self.named_parameters():
            # 兼容 DeepSpeed ZeRO 3 的参数分片
            if hasattr(param, "ds_numel") and param.numel() == 0:
                num_params = param.ds_numel
            else:
                num_params = param.numel()
            # 兼容 4-bit 量化模型的参数计算
            if "4bit" in param.__class__.__name__.lower():
                num_params = num_params * 2
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param


    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: "
            f"{trainable_params:,d} || all params: "
            f"{all_param:,d} || trainable%: "
            f"{100 * trainable_params / all_param:.4f}"
        )

    def print_model_structure(self):
        """打印完整的模型结构，包括注入的 LoRA 层。"""
        print("=" * 60)
        print("RefSegEncoderDecoder Full Model Structure:")
        # 直接打印 self 对象
        # PyTorch/HuggingFace 的 repr() 会递归打印所有子模块
        print(self)
        print("=" * 60)

    def get_image_positional_embeddings(self, size):
        target_device = self.backbone.shared_image_embedding.positional_embedding.device
        target_dtype  = self.backbone.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        positional_embedding = self.backbone.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width


    def extract_feat(self, inputs, data_samples: SampleList, return_attns: bool = False) -> List[Tensor]:
        """
        特征融合流程：
            1. 提取 SAM 高分辨率几何特征
            2. 提取 SigLIP 视觉特征 & POI 文本语义特征
            3. 细粒度局部融合 (Cross-modal Interaction)
            4. SAM 提示工程 (BBox + Scene Token)
            5. 解码输出分割与分类特征
        """
        # print("data_samples: extract_feat", data_samples)
        # print("inputs: extract_feat: ", inputs)

        ####################################### SAM 视觉特征提取 (侧重高分辨率几何结构)
        x_sam = normalize(inputs, mean=self.backbone.processor.image_mean, std=self.backbone.processor.image_std) # 按照 SAM 预处理规范对输入图像进行归一化
        # print("x_sam: ", x_sam)
        # print("x_sam.shape: ", x_sam.shape) # torch.Size([1, 3, 1024, 1024])
        sam_visual_feat = self.backbone(x_sam)['last_hidden_state'] # 通过 SAM 主干网络提取高分辨率视觉特征
        # print("sam_visual_feat: ", sam_visual_feat.shape) # torch.Size([1, 256, 64, 64])
        ####################################### SAM 视觉特征提取 (侧重高分辨率几何结构)


        # ####################################### SigLIP 视觉特征提取 (侧重深层语义感知)
        # # 图像预处理：按照 SigLIP 的规范进行归一化
        # x_clip = normalize(inputs, mean=self.clip_vision_encoder.processor.image_mean, std=self.clip_vision_encoder.processor.image_std)
        # # 缩放图像：插值到 SigLIP 要求的固定输入尺寸
        # x_clip = F.interpolate(x_clip, size=list(self.clip_vision_encoder.processor.size.values()), mode='bicubic', align_corners=False, antialias=True)
        # # 提取特征：获得原始序列格式的 Patch Token
        # clip_visual_feat = self.clip_vision_encoder(x_clip)['last_hidden_state']
        # # print("clip_visual_feat: ", clip_visual_feat.shape) # torch.Size([1, 256, 1024])
        # # 自动处理 CLS Token 并还原空间维度
        # num_tokens = clip_visual_feat.shape[1]
        # grid_size = int(num_tokens ** 0.5)
        # if grid_size * grid_size != num_tokens:
        #     # 存在 CLS Token (通常在第 0 位)，进行切片
        #     print("针对 clip_visual_feat 执行 CLS Token 切片")
        #     clip_visual_feat = clip_visual_feat[:, 1:, :]
        #     grid_size = int(clip_visual_feat.shape[1] ** 0.5)
        # clip_visual_feat = einops.rearrange(clip_visual_feat, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)
        # # print("clip_visual_feat: ", clip_visual_feat.shape) # torch.Size([1, 1024, 16, 16])
        # ####################################### SigLIP 视觉特征提取 (侧重深层语义感知)

        ####################################### resnet-50 视觉特征提取
        # 1. 图像归一化 # 使用 ImageNet 统计量（ResNet 预训练假设）
        x_res = normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 2. 图像缩放 # ResNet 通常以 224×224 为标准输入
        x_res = F.interpolate(x_res, size=(512, 512), mode='bicubic', align_corners=False)
        # 3. 特征提取 # ResNet Vision Encoder 返回 (B, N, C)
        res_visual_feat = self.clip_vision_encoder(x_res)['last_hidden_state']
        # print("clip_visual_feat: ", clip_visual_feat.shape) # torch.Size([1, 256, 1024])
        # 4. 直接赋值 (ResNet输出已经是 B C H W 格式，不需要 rearrange)
        clip_visual_feat = res_visual_feat
        # print("clip_visual_feat: ", clip_visual_feat.shape) # torch.Size([1, 1024, 16, 16])
        ####################################### resnet-50 视觉特征提取


        # Visual Token Dropout
        if self.training:
            B, C, H, W = clip_visual_feat.shape
            num_tokens = H * W
            keep_prob = 0.7  # 可调参数，建议 0.6~0.8
            mask = torch.rand(B, num_tokens, device=clip_visual_feat.device) < keep_prob
            mask = mask.float().view(B, 1, H, W)  # [B, 1, H, W]
            clip_visual_feat = clip_visual_feat * mask


        ####################################### POI 地理文本语义编码
        text_list = [ds.get('poi_prompt') for ds in data_samples]
        text_dict = self.clip_text_encoder.processor(text_list, return_tensors='pt', padding=True, truncation=True, max_length=64)
        text_dict = {k: v.to(inputs.device) for k, v in text_dict.items()}
        clip_text_feat = self.clip_text_encoder(**text_dict)
        clip_text_feat_pooler = clip_text_feat['pooler_output'] # 全局功能分类 / Scene token
        clip_text_feat = clip_text_feat['last_hidden_state']    # BXSeqLenX1152 (局部 Token)
        # print(f"text_dict keys: {text_dict.keys()}")
        ####################################### POI 地理文本语义编码


        # ############################# 细粒度局部融合 (siglip视觉特征与POI语义特征的跨模态交互) （双路激活机制融合）
        # # 归一化以计算余弦相似度
        # v_norm = F.normalize(clip_visual_feat, dim=1)
        # t_tokens_norm = F.normalize(clip_text_feat, dim=2)
        # t_pooler_norm = F.normalize(clip_text_feat_pooler, dim=1)
        # # --- 支路 A: 全局激活 (Global Activation) ---
        # # 计算全图对 POI 总体描述的响应程度，作为空间 Gate
        # global_sim = einops.einsum(v_norm, t_pooler_norm, 'b c h w, b c -> b h w')
        # global_gate = torch.sigmoid(global_sim * self.clip_text_encoder.logit_scale.exp() + self.clip_text_encoder.logit_bias)
        # global_gate = global_gate.unsqueeze(1)  # [B, 1, H, W]
        # # --- 支路 B: 局部激活 (Local Activation) ---
        # # 计算像素与具体 POI 词汇的相关性
        # local_logits = einops.einsum(v_norm, t_tokens_norm, 'b c h w, b t c -> b t h w')
        # local_attn = torch.softmax(local_logits * self.clip_text_encoder.logit_scale.exp(), dim=1)
        # # 将文本语义投影到视觉空间
        # local_text_at_pixel = einops.einsum(local_attn, clip_text_feat, 'b t h w, b t c -> b c h w')
        # # --- 最终融合 ---
        # # 局部语义注入 * 全局激活门控 + 原始视觉
        # # self.text_to_vis_proj 是一个 1x1 卷积层，将文本维度转为视觉维度
        # enhanced_semantic_feat = self.text_to_vis_proj(local_text_at_pixel) * global_gate
        # local_fused_feat       = clip_visual_feat + self.poi_scale * enhanced_semantic_feat # 使用可缩放因子
        # ############################# 细粒度局部融合 (siglip视觉特征与POI语义特征的跨模态交互)


        ############################# siglip视觉特征与POI语义特征的拼接融合
        # 1. 准备数据
        B, C, H, W       = clip_visual_feat.shape
        # clip_text_feat_pooler shape: [B, C]
        # 2. 空间广播 (Broadcasting)
        # 将全局文本向量扩展为与图像一样的 HxW 网格
        text_global      = clip_text_feat_pooler.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        text_expanded    = text_global.expand(-1, -1, H, W)  # [B, C, H, W]
        # 3. 通道拼接 (Concatenation)
        # 在通道维度 (dim=1) 进行拼接
        # 输入维度变为 C + C = 2C
        cat_feat         = torch.cat([clip_visual_feat, text_expanded], dim=1)  # [B, 2*C, H, W]
        # 4. 降维与归一化 (Projection + GroupNorm)
        # 将 2C 维度的特征压缩回 C 维度，并经过 GN 处理
        local_fused_feat = self.simple_concat_proj(cat_feat)  # [B, C, H, W]
        ############################# siglip视觉特征与POI语义特征的拼接融合


        ############################# SAM 掩膜生成与特征对齐
        # 位置编码：生成适配 SAM 维度的空间坐标嵌入
        batch_size = inputs.size(0) # 获取当前 batch 的样本数量
        image_positional_embeddings = self.get_image_positional_embeddings(sam_visual_feat.size(2)) # 生成与 SAM 图像特征分辨率匹配的位置编码
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1) # 将位置编码扩展到 batch 维度
        # print("image_positional_embeddings:", image_positional_embeddings.shape) # torch.Size([1, 256, 64, 64])

        # 提取 BBox (从 GT Mask 转化而来)
        target_bboxes = [] # 初始化用于存储每个样本目标 bbox 的列表
        for ds in data_samples: # 遍历每个样本以提取或构造 bbox
            if hasattr(ds, 'gt_instances') and len(ds.gt_instances.bboxes) > 0:
                bbox = ds.gt_instances.bboxes[0] # 取第一个 bbox (单实例设定)
            else:
                h, w = inputs.shape[-2:]
                bbox = torch.tensor([0, 0, w, h], device=inputs.device, dtype=torch.float)
            target_bboxes.append(bbox)
        target_bboxes = torch.stack(target_bboxes) # 将 bbox 列表堆叠为 batch tensor
        # 若为二维 bbox 张量，则补充 instance 维度
        if target_bboxes.dim() == 2:
            target_bboxes = target_bboxes.unsqueeze(1)

        #全局语义调制机制
        gate_values     = self.context_gate(clip_text_feat_pooler).unsqueeze(-1).unsqueeze(-1)  # [B, 256, 1, 1]
        sam_visual_feat = sam_visual_feat * gate_values

        # SAM Prompt Encoding
        sparse_emb_box, dense_prompt = self.sam_prompt_encoder(
            input_points = None,
            input_labels = None,
            input_boxes  = target_bboxes,
            input_masks  = None
        )
        # print("sparse_embeddings_box: ", sparse_embeddings_box.shape) # torch.Size([1, 1, 2, 256])

        # 掩膜解码：利用 SAM 解码器生成最终的低分辨率预测图
        low_res_masks, _, _ = self.sam_mask_decoder(
            image_embeddings            = sam_visual_feat,
            image_positional_embeddings = image_positional_embeddings,
            sparse_prompt_embeddings    = sparse_emb_box,
            dense_prompt_embeddings     = dense_prompt,
            multimask_output            = False,
        )
        # print("low_res_masks:", low_res_masks.shape) # torch.Size([1, 1, 1, 256, 256])
        sam_logits = low_res_masks.squeeze(1)

        # # 是否返回注意力权重
        # if return_attns:
        #     # attns_dict = {
        #     #     'local_attns': list(attn_probs.detach().cpu()),
        #     #     'sam_mask': list(sam_pred_mask.detach().cpu())
        #     # }
        #     return target_geo_feat, target_sem_feat, clip_text_feat_pooler, sam_logits, attns_dict
        return sam_visual_feat, local_fused_feat, clip_text_feat_pooler, sam_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """训练阶段入口"""
        # 提取原始特征
        sam_high_res, local_feat, global_feat, sam_mask = self.extract_feat(inputs, data_samples)
        losses = dict()

        # 分割损失
        bg_logits = torch.zeros_like(sam_mask)
        sam_mask_2ch = torch.cat([bg_logits, sam_mask], dim=1)
        feat_tuple_seg = (sam_high_res, local_feat, global_feat, sam_mask_2ch)
        losses.update(self.seg_head.loss(feat_tuple_seg, data_samples, self.train_cfg))

        # # 对比学习损失 (CLIP-like Alignment)
        # # 获取用于对齐的全局特征
        # v_alignment = local_feat.mean(dim=[-2, -1])  # 空间平均池化
        # t_alignment = global_feat
        # v_emb = F.normalize(self.v_proj_head(v_alignment), dim=-1)
        # t_emb = F.normalize(self.t_proj_head(t_alignment), dim=-1)
        # # 计算相似度矩阵
        # ctr_logits = torch.matmul(v_emb, t_emb.t()) * self.logit_scale.exp()
        # ctr_labels = torch.arange(ctr_logits.size(0), device=inputs.device)
        # loss_ctr = (F.cross_entropy(ctr_logits, ctr_labels) + F.cross_entropy(ctr_logits.t(), ctr_labels)) / 2
        # losses['loss_aux_contrastive'] = loss_ctr * self.ctr_loss_weight

        # 分类任务
        # sam_mask_for_cls = sam_mask * 0.1 + sam_mask.detach() * 0.9 # 允许 10% 的梯度回传到 SAM 分割网络，确保分割出的形状对推理有贡献
        feat_tuple_cls = (sam_high_res, local_feat, global_feat, sam_mask.detach())
        losses.update(self.cls_head.loss(feat_tuple_cls, data_samples, self.train_cfg))
        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList):
        """推理阶段入口"""
        # print("inputs: predict: ", inputs)
        # 获取图像元信息
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # 构造默认元数据
            batch_img_metas = [dict(ori_shape=inputs.shape[2:],img_shape=inputs.shape[2:],pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])] * inputs.shape[0]

        # 1. 前向特征提取
        sam_high_res, local_feat, global_feat, sam_mask_logits = self.extract_feat(inputs, data_samples)
        feat_tuple = (sam_high_res, local_feat, global_feat, sam_mask_logits)

        # 构造 2通道用于分割输出
        bg_logits = torch.zeros_like(sam_mask_logits)
        sam_mask_2ch = torch.cat([bg_logits, sam_mask_logits], dim=1)

        # 2. 分类预测 (Head 内部做融合)
        cls_scores = self.cls_head.forward(feat_tuple)

        # 3. 结果打包
        for i, data_sample in enumerate(data_samples):
            ori_h, ori_w = batch_img_metas[i]['ori_shape'][:2]

            # 将 2通道 Logits 插值回原图尺寸
            mask_logit_resized = F.interpolate(
                sam_mask_2ch[i].unsqueeze(0),
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # 存入 PixelData。此时 Metric 执行 argmax(dim=0) 就能得到正确的 0/1 掩码
            data_sample.pred_sem_seg = PixelData(data=mask_logit_resized)
            data_sample.pred_logits = LabelData(score=cls_scores[i])

        return data_samples


@MODELS.register_module()
class SamSupervisionHead(BaseModule):
    """专门为 SAM 分支设计的监督头，负责二值化建筑分割任务的损失计算"""
    def __init__(
            self,
            num_classes: int  = 2,   # 类别数：默认为2（背景、建筑）
            out_channels: int = 1,   # 输出通道：对于多类分割通常与类别一致
            threshold: float  = 0.5, # 阈值：用于推理阶段的二值化判断
            ignore_index      = 255, # 忽略索引：计算 Loss 时忽略的像素点（如填充区域）
            loss_decode=dict,
            sampler=None,            # 采样器：用于处理正负样本不平衡
            align_corners=False,     # 缩放对齐方式
            init_cfg=None,           # 初始化配置
    ):
        super().__init__(init_cfg=init_cfg) # 调用基类初始化
        self.num_classes = num_classes      # 存储类别数量
        self.out_channels = out_channels    # 存储输出通道数
        self.threshold = threshold          # 存储预测阈值
        self.ignore_index = ignore_index    # 存储忽略索引
        self.align_corners = align_corners  # 存储对齐配置

        # 动态构建损失函数 (支持单个字典或多个损失字典列表)
        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode) # 构建单个损失模块
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:                     # 若为列表则构建模块列表
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        # 如果配置了像素采样器则构建，否则设为空
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

    def forward(self, inputs):
        """
            inputs 来自 extract_feat:
            (target_geo_feat, target_sem_feat, clip_text_feat_pooler, sam_logits)
        """
        if isinstance(inputs, tuple):
            # 解包元组，提取第4个元素 (sam_logits) # target_geo_feat, target_sem_feat, clip_text_feat_pooler, sam_logits
            return inputs[3]
        return inputs

    def loss(self, inputs, batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        """训练入口，负责获取预测值并开启 Loss 计算流程。"""
        seg_logits = self.forward(inputs)
        return self.loss_by_feat(seg_logits, batch_data_samples)        # 调用核心计算函数

    def loss_by_feat(self, seg_logits: Tensor, batch_data_samples: SampleList) -> dict:
        """计算二值分割 Loss。将预测值与 GT 对齐并计算具体的数值。"""
        # print("batch_data_samples:", batch_data_samples)
        # --- 1. 提取并处理标签 (GT) ---
        seg_label = self._stack_batch_gt(batch_data_samples)
        # 尺寸对齐：将 GT 缩放到与模型输出尺寸一致 (例如从 1024 缩放到 256)
        seg_label_down = resize(
            input=seg_label.float(),    # 转为浮点以支持插值
            size=seg_logits.shape[2:],  # 目标尺寸为预测图的高宽
            mode='nearest',             # 使用最近邻插值，保证标签值依然是 0 或 1
            align_corners=None
        ).squeeze(1).long()             # 移除通道维并转回长整型索引，供交叉熵使用

        # --- 2. 准备预测值 ---
        # 此时 seg_logits 已经是 [B, 2, H, W]，直接使用，无需再 cat
        standard_logits = seg_logits.float()

        # --- 3. 计算多损失函数 ---
        loss = dict()
        # 循环计算损失：处理可能存在的多个 Loss 函数 (如 CE + Dice)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            # 计算预测值与 GT 之间的偏差
            loss_value = loss_decode(standard_logits, seg_label_down, ignore_index=self.ignore_index)
            # 将该损失项存入结果字典
            loss[loss_decode.loss_name] = loss_value
        # 辅助指标：计算预测掩码的准确率(Accuracy)
        loss['acc_sam_seg'] = accuracy(standard_logits, seg_label_down, ignore_index=self.ignore_index)
        return loss

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """从数据样本 DataSample 中精准提取建筑分割的标签图。"""
        gt_semantic_segs = []
        for data_sample in batch_data_samples:
            # 获取 GT Mask (通常是 [1, H, W]) # 注意：RefSegDataset 应该确保 gt_sem_seg 是二值的 (0/1)
            if hasattr(data_sample, 'gt_building_seg'): # 确认gt_building_seg字段是建筑分割字段，需要将segmentation保存到json数据中
                gt_seg = data_sample.gt_building_seg.data
            elif hasattr(data_sample, 'gt_sem_seg'):
                gt_seg = data_sample.gt_sem_seg.data # 兼容性写法：如果以后用回标准数据集，也能跑
            else:
                raise AttributeError("DataSample 中找不到 'gt_function_seg' 或 'gt_sem_seg' 属性！")
            gt_semantic_segs.append(gt_seg)
        return torch.stack(gt_semantic_segs, dim=0) # 将列表中的多个样本 Tensor 堆叠为一个 Batch Tensor


@MODELS.register_module()
class ReasoningClsHead(BaseModule):
    """
        功能：建筑功能推理头。
        逻辑：通过功能原型（Query）在掩膜限制的视觉证据（Key/Value）中寻找分类线索。
    """
    def __init__(self,
                 in_channels   = 1024,  # 输入维度：适配 SigLIP 的 1024 维
                 sam_channels  = 256,   # 显式定义 SAM 的输入维度
                 num_classes   = 3,     # 为 3 类城市功能
                 num_heads     = 8,
                 dropout_ratio = 0.1,
                 loss_cls      = None,
                 norm_cfg      = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls_func = MODELS.build(loss_cls)  # 实例化分类损失函数

        # 维度投影：将 SAM 特征对齐到 SigLIP 维度
        self.sam_proj      = nn.Conv2d(sam_channels, in_channels, kernel_size=1)
        # 特征归一化
        self.evidence_norm = nn.LayerNorm(in_channels) # 功能原型：3 个可学习的向量，代表每类功能的知识先验
        # 融合层：整合几何(SAM)与语义(SigLIP)
        self.fusion_conv   = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(32, in_channels),
            nn.GELU()
        )
        # 动态原型生成器：将 POI 语义注入静态知识
        self.proto_adapter = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, num_classes * in_channels)
        )

        # 核心推理引擎：Cross-Attention
        self.function_prototypes = nn.Parameter(torch.randn(num_classes, in_channels) * 0.02)
        self.reasoning_engine    = nn.MultiheadAttention(
            embed_dim   = in_channels,
            num_heads   = num_heads,
            batch_first = True,
            dropout     = dropout_ratio
        )

        # 残差分支投影
        self.feat_to_proto_proj = nn.Linear(in_channels, in_channels)
        # 初始化权重：使初期 POI 注入保持中性
        self.proto_adapter[-1].weight.data.zero_()
        self.proto_adapter[-1].bias.data.zero_()

        # 最终决策层
        self.consistency_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Linear(in_channels // 2, 1) # 输出标量分值
        )

    def forward(self, inputs):
        """核心逻辑：掩膜引导的特征提取 -> 跨模态推理 -> 决策融合"""
        # 对应 RefSeg 传进来的 4 个变量 (去掉了已经不存在的 GAT)
        sam_v, local_s, poi_feat, sam_mask = inputs
        B                                  = sam_v.size(0) # batchsize
        target_sz                          = sam_v.shape[-2:]

        ############################## 1. 空间与维度对齐
        sam_v_proj = self.sam_proj(sam_v)
        local_s_up = F.interpolate(local_s, size=target_sz, mode='bilinear', align_corners=False) # 统一缩放到 SAM 特征图尺寸 (64x64)
        fused_v    = self.fusion_conv(torch.cat([sam_v_proj, local_s_up], dim=1))

        ############################## 2. 构建精确证据池
        # 直接使用 SAM 输出的掩膜进行插值
        soft_mask       = F.interpolate(torch.sigmoid(sam_mask), size=target_sz, mode='bilinear', align_corners=False)
        # 将特征平铺为序列格式
        visual_evidence = self.evidence_norm(einops.rearrange(fused_v, 'b c h w -> b (h w) c'))
        # 将掩膜平铺为权重序列
        mask_weight     = einops.rearrange(soft_mask, 'b 1 h w -> b (h w) 1')
        # 应用掩膜加权。保留一个小偏置 (如 0.1) 是为了防止非建筑区域特征被完全清零导致的梯度断裂
        final_evidence  = visual_evidence * (mask_weight + 0.1)

        ############################## 3. 跨模态推理
        # 生成动态原型
        dynamic_bias      = self.proto_adapter(poi_feat).view(B, self.num_classes, -1) # 让 POI 语义决定分类器的“期望”
        prototypes        = self.function_prototypes.unsqueeze(0) + dynamic_bias       # 扩展原型：将 3 个功能原型扩展到当前 Batch 的每个样本中
        # Reasoning: 用原型去匹配掩膜内的视觉证据
        updated_protos, _ = self.reasoning_engine(prototypes, final_evidence, final_evidence) # 核心推理：Query(原型) 在 Key/Value(视觉证据池) 中进行匹配

        ############################## 4. 全局上下文补偿 (残差连接)
        # 计算掩膜内区域的全局平均特征，作为残差补充
        denom                  = mask_weight.sum(1) + 1e-6
        global_visual_evidence = (visual_evidence * mask_weight).sum(1) / denom
        updated_protos         = updated_protos + self.feat_to_proto_proj(global_visual_evidence).unsqueeze(1)

        ############################## 5. 计算最终得分
        logits = self.consistency_head(updated_protos).squeeze(-1)
        return logits

    def loss(self, inputs, batch_data_samples, train_cfg=None) -> dict:
        """分类损失计算入口"""
        cls_score = self.forward(inputs) # 获得预测分数
        return self.loss_by_feat(cls_score, batch_data_samples) # 调用指标计算逻辑

    def loss_by_feat(self, cls_score, batch_data_samples) -> dict:
        """计算交叉熵损失并监控准确率"""
        # 提取真值标签并迁移到 GPU
        labels = self._stack_batch_gt(batch_data_samples).to(cls_score.device)
        losses = dict()
        # 计算主要的分类损失
        losses['loss_function_cls'] = self.loss_cls_func(cls_score.float(), labels)
        # 监控指标：计算当前 Batch 的分类准确率
        with torch.no_grad():
            pred_label = cls_score.argmax(dim=1)
            acc = (pred_label == labels).float().mean() * 100.0
            losses['acc_cls'] = acc
        return losses

    def _stack_batch_gt(self, batch_data_samples):
        """从样本元数据中提取分类真值"""
        gt_labels = []
        for data_sample in batch_data_samples:
            gt_label_data = data_sample.gt_label # 获取 LabelData 对象
            # 兼容 LabelData 对象和原始 Tensor # 注意：在 PackMultiTaskInputs 里是这样存的: gt_label_data.label = label_tensor
            if hasattr(gt_label_data, 'label'):
                label = gt_label_data.label
            else:
                label = gt_label_data # 兼容性处理：万一它不是 LabelData 而是直接存的 Tensor
            # 确保它是 Tensor
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            gt_labels.append(label)
        return torch.stack(gt_labels).squeeze(1).to(dtype=torch.long)


@MODELS.register_module()
class RefSegSiglipTextModel(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='lcybuaa1111/Git-RSCLIP',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path).tokenizer
        model = AutoModel.from_pretrained(model_name_or_path)
        self.model = model.text_model
        self.config = self.model.config
        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias
        self.model.is_init = True
        self.logit_scale.is_init = True
        self.logit_bias.is_init = True
    def init_weights(self):
        pass
    def forward(self, *args, **kwargs):
        results = self.model(*args, **kwargs)
        return results


@MODELS.register_module()
class RefSegResNetVisionModel(BaseModule):
    """
        替代 SAM/SigLIP Vision Encoder 的 ResNet-50 模型。
        保持Transformer 的输出格式 (B, N, C)。
    """
    def __init__(
            self,
            model_name_or_path: str,
            local_ckpt_path: str,
            target_hidden_size: int = 1024,  # 显式定义输出维度，通常 SigLIP Large为1024或1152
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.config = type('', (), {})()
        self.config.hidden_size = target_hidden_size
        self.model_name_or_path = model_name_or_path
        # 1. 处理器加载 (用于获取 input_size，保持逻辑一致)
        downloaded_path = snapshot_download(model_name_or_path)
        self.processor  = AutoProcessor.from_pretrained(downloaded_path)
        # 3. 构建 ResNet-50 骨干
        self.model = resnet50(weights=None)
        self.model.avgpool = nn.Identity()  # 移除全局池化
        self.model.fc      = nn.Identity()  # 移除全连接层
        # 4. 加载 ResNet 权重
        self._load_and_clean_weights(local_ckpt_path)
        # 5. 维度投影层 (2048 -> target_hidden_size)
        self.output_proj = nn.Conv2d(2048, target_hidden_size, kernel_size=1)
        self.model.is_init = True
    def _load_and_clean_weights(self, local_ckpt_path):
        """加载权重并清理掉不必要的 'fc' 层权重。"""
        # 路径处理
        ckpt_path = local_ckpt_path
        print(f"[RefSegResNet] Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 兼容处理：有些ckpt直接是dict，有些在['model']里
        state_dict = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        # 过滤掉 fc 层的权重，避免加载报错
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"[RefSegResNet] Load status: {msg}")
    def init_weights(self):
        pass
    def forward(self, pixel_values, return_2d_feature=True, **kwargs):
        """
        Args:
            pixel_values: (B, 3, H, W) 输入图像
        Returns:
            BaseModelOutputWithPooling: 包含 last_hidden_state 和 pooler_output
        """
        # 1. ResNet 前向传播
        x = self.model.conv1(pixel_values)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        feats = self.model.layer4(x)  # 输出形状: (B, 2048, H_feat, W_feat)
        # 2. 通道降维 (2048 -> 1024)
        x = self.output_proj(feats)  # (B, 1024, H_feat, W_feat)
        if return_2d_feature:
            # 直接返回空间特征，方便后续做分割任务
            return {'last_hidden_state': x}
        else:
            # 兼容 Transformer 接口 (如果必须的话)
            last_hidden_state = x.flatten(2).transpose(1, 2)
            return {'last_hidden_state': last_hidden_state}
# @MODELS.register_module()
# class RefSegSiglipVisionModel(BaseModule):
#     def __init__(
#             self,
#             model_name_or_path: str='lcybuaa1111/Git-RSCLIP',
#             init_cfg=None
#     ):
#         super().__init__(init_cfg=init_cfg)
#         self.model_name_or_path = model_name_or_path
#         model_name_or_path = snapshot_download(model_name_or_path)
#         self.processor = AutoProcessor.from_pretrained(model_name_or_path).image_processor
#         self.model = AutoModel.from_pretrained(model_name_or_path).vision_model
#         self.config = self.model.config
#         self.model.is_init = True
#     def init_weights(self):
#         pass
#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)


@MODELS.register_module()
class RefSegSamVisionEncoder(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='KyanChen/sam-vit-base',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        self.processor = SamProcessor.from_pretrained(model_name_or_path).image_processor
        model = SamModel.from_pretrained(model_name_or_path)
        self.model = model.vision_encoder
        self.config = self.model.config
        self.shared_image_embedding = model.shared_image_embedding
        self.model.is_init = True
        self.shared_image_embedding.is_init = True
    def init_weights(self):
        pass
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

@MODELS.register_module()
class RefSegSamPromptEncoder(BaseModule):
    def __init__(
            self,
            model_name_or_path: str = 'KyanChen/sam-vit-base',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        model = SamModel.from_pretrained(model_name_or_path)
        self.model = model.prompt_encoder
        self.hidden_size = self.model.hidden_size
        self.model.is_init = True
    def init_weights(self):
        pass
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

@MODELS.register_module()
class RefSegSamMaskDecoder(BaseModule):
    def __init__(
            self,
            model_name_or_path: str = 'KyanChen/sam-vit-base',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        model = SamModel.from_pretrained(model_name_or_path)
        self.model = model.mask_decoder
        self.hidden_size = self.model.hidden_size
        self.model.is_init = True
    def init_weights(self):
        pass
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)



if __name__ == '__main__':
    siglip_vision_encoder = RefSegSiglipTextModel(model_name_or_path='lcybuaa1111/Git-RSCLIP')
    print(siglip_vision_encoder.model.is_init)