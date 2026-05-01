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


from mmengine.optim import AmpOptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from mmengine.visualization import LocalVisBackend, WandbVisBackend
from torch.optim import AdamW
from mmseg.visualization import SegLocalVisualizer
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook)
from mmengine.optim.scheduler.lr_scheduler import PolyLR, LinearLR, CosineAnnealingLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from mmseg.engine import SegVisualizationHook
from mmcv.transforms.loading import LoadImageFromFile
from mmengine.dataset.sampler import DefaultSampler
from rsris import RefSegClsMetric
from rsris.datasets.refdataset import RefSegDataset, LoadSegAnnotations, NewResize, ToTensor, PackMultiTaskInputs
from rsris.models.models import RefSegEncoderDecoder, RefSegSiglipVisionModel, RefSegSiglipTextModel, RefSegSamVisionEncoder, RefSegSamPromptEncoder, RefSegSamMaskDecoder


default_scope = 'mmseg'
custom_imports = dict(imports=['rsris'], allow_failed_imports=False)


work_dir             = f'./work_dirs/THREE_CLS'
data_root            = f'./dataset/JL_Images'
# data_root            = f'/XYFS01/HDD_POOL/njnu_ynwen/njnu_ynwen_1/JL_Images'
building_labels_root = f'./dataset/building_labels'
# building_labels_root = f'/XYFS01/HDD_POOL/njnu_ynwen/njnu_ynwen_1/building_labels'


batch_size   = 1
max_epochs   = 100
val_interval = 1


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type=LocalVisBackend),
    # dict(type=WandbVisBackend, init_kwargs=dict(project='RSRefSeg', group='RSRefSeg-b', name=work_dir.split('/')[-1]))
]
visualizer = dict(type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'


# 断点续训
# 开启断点续训模式，这会告诉 Runner 读取 checkpoint 中的 meta 信息（包括 epoch 计数）
resume    = False
# resume    = True
# load_from = './work_dirs/concat-fusion-0116/epoch_23.pth'


# 指定你要接着训练的权重路径 # 通常用于“迁移学习”或“微调”（即加载权重但重置 Epoch 从 0 开始，且不加载优化器状态）。既
init_from = None
# init_from = dict(
#     type='Pretrained',
#     checkpoint='./work_dirs/PHASE2_CLS_ONLY/epoch_16.pth' #
# )


train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=val_interval)
val_cfg   = dict(type=ValLoop)
test_cfg  = dict(type=TestLoop)


default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=True,
        interval=val_interval,
        max_keep_ckpts=5, # 保存最近的5个模型
        save_last=True,
        save_best='RefSegClsMetric/Function_F1_Macro',
        rule='greater',
        # filename_tmpl='epoch_{}', # 分布式计算 设置
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook),
    early_stop=dict(
        type='EarlyStoppingHook',
        monitor='RefSegClsMetric/Function_F1_Macro',
        rule='greater',  # 指标越大越好
        patience=20,     # 容忍多少个验证周期不提升就停止
        min_delta=0.0001 # 允许的最小提升幅度
    )
)

crop_size = (1024, 1024)
# crop_size = (512, 512)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadSegAnnotations),
    dict(type=NewResize,
         scale=crop_size,
         keep_ratio=False,
         resize_labels=True,  # 训练时要对标签同步进行缩放
         ),
    dict(type=ToTensor, keys=['gt_building_seg']),
    dict(type=PackMultiTaskInputs,
         meta_keys=('poi_text', 'poi_prompt',
                    'img_path', 'seg_map_path', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'reduce_zero_label')
         )
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadSegAnnotations),
    dict(type=NewResize,
         scale=crop_size,
         keep_ratio=False,
         resize_labels=False, # 验证/测试时不需要对标签进行缩放
         ),
    dict(type=ToTensor, keys=['gt_building_seg']),
    dict(type=PackMultiTaskInputs,
         meta_keys=('poi_text', 'poi_prompt',
                    'img_path', 'seg_map_path', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'reduce_zero_label')
         )
]

# dataset settings
dataset_type       = RefSegDataset
num_workers        = 8
persistent_workers = True

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=True),
    # sampler=dict(
    #     type="ClassAwareSampler",
    #     num_samples_per_class=1,
    # ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        building_labels_root=building_labels_root,
        ann_file='datainfo/train.jsonl',
        pipeline=train_pipeline),
        drop_last=True,  # 如果最后一个 Batch 的数量不足设定的 batch_size，就直接丢弃，不进行训练。
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        building_labels_root=building_labels_root,
        ann_file='datainfo/val.jsonl',
        pipeline=test_pipeline,
        test_mode=True),
)
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        building_labels_root=building_labels_root,
        ann_file='datainfo/test.jsonl',
        pipeline=test_pipeline,
        test_mode=True),
)


# 设置metric
# 设置评估指标：同时评估分割和分类
val_evaluator = [
    # dict(type=RefSegIoUMetric),  # 分割指标（建筑/非建筑）
    dict(type=RefSegClsMetric)     # 分类指标（1类非建筑、5类建筑功能）
]
test_evaluator = val_evaluator


# model settings
# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

data_preprocessor = dict(
    type       = 'SegDataPreProcessor',
    mean       = [0, 0, 0],
    std        = [255., 255., 255.],  # normalize the image in the model internally
    bgr_to_rgb = True,
    pad_val    = 0,
    seg_pad_val= 255,
    size       = crop_size,
)

model = dict(
    type=RefSegEncoderDecoder,
    data_preprocessor=data_preprocessor,
    init_cfg=init_from,
    norm_cfg=norm_cfg,

    lora_cfg=dict(
        backbone=dict(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['qkv', 'proj', 'lin1', 'lin2', 'neck.conv1', 'neck.conv2']
        ),
        clip_vision_encoder=dict(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            # target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj']
            target_modules=['q_proj', 'v_proj']
        ),
        clip_text_encoder=dict(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj', 'mlp.fc1', 'mlp.fc2']
            # target_modules=['q_proj', 'v_proj']
        ),
    ),

    backbone=dict(
        type=RefSegSamVisionEncoder,
        model_name_or_path='KyanChen/sam-vit-base'
    ),
    clip_vision_encoder=dict(
        type=RefSegSiglipVisionModel,
        model_name_or_path='lcybuaa1111/Git-RSCLIP',
    ),
    clip_text_encoder=dict(
        type=RefSegSiglipTextModel,
        model_name_or_path='lcybuaa1111/Git-RSCLIP',
    ),
    sam_prompt_encoder=dict(
        type=RefSegSamPromptEncoder,
        model_name_or_path='KyanChen/sam-vit-base',
    ),
    sam_mask_decoder=dict(
        type=RefSegSamMaskDecoder,
        model_name_or_path='KyanChen/sam-vit-base',
    ),

    # 对应 SamSupervisionHead
    seg_head=dict(
        type='SamSupervisionHead',
        num_classes  = 2,  # 二分类 (背景 vs 建筑)
        out_channels = 1,
        threshold    = 0.5,
        loss_decode  = [
            dict(
                type        = 'mmseg.CrossEntropyLoss',
                use_sigmoid = False,
                loss_weight = 1.0,
                loss_name   = 'loss_ce'
                ),
            dict(
                type        = 'mmseg.LovaszLoss',
                loss_type   = 'multi_class',
                classes     = [1], # 重点：只关注前景
                per_image   = False,
                reduction   = 'none',
                loss_weight = 1.0,
                loss_name   = 'loss_lovasz'
            )
        ]
    ),

    # 对应 ReasoningClsHead
    cls_head=dict(
        type='ReasoningClsHead',
        in_channels   = 1024,  # 对应 SigLIP 的 hidden dim
        num_classes   = 3,     # 城市功能类别数
        num_heads     = 8,     # 新增：推理引擎的注意力头数
        dropout_ratio = 0.1,
        norm_cfg      = norm_cfg,
        loss_cls      = dict(
            type='mmseg.CrossEntropyLoss',
            use_sigmoid  = False,  # 推理头输出 3 类得分，通常使用标准 CE
            loss_weight  = 3.0,
            # class_weight = [0.6634, 0.6634, 2.4496, 1.1788, 0.7178, 0.6636, 0.6634],
            loss_name='loss_function_cls'
        ),
        # loss_cls=dict(
        #     type='mmseg.FocalLoss',
        #     use_sigmoid=True,      # 必须为 True，该实现仅支持 Sigmoid
        #     gamma=2.0,             # 默认值，调节难易样本权重，2.0 是经典设定
        #     alpha=0.25,            # 调节正负样本平衡，因为是多标签风格，正样本稀疏，0.25 较常用
        #     loss_weight=3.0,       # 分类任务权重
        #     class_weight=[0.6, 0.6, 2.4, 1.2, 0.6, 0.6, 0.6],
        #     loss_name='loss_function_cls'
        # ),
    ),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


base_lr = 0.0001
find_unused_parameters=True

param_scheduler = [
    dict(type=LinearLR, start_factor=0.01, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(
        type          = CosineAnnealingLR,
        T_max         = max_epochs,
        by_epoch      = True,
        begin         = 5,
        eta_min_ratio = 0.01,
        end           = max_epochs
    ),
]

### AMP training config
runner_type = 'Runner'
optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype='bfloat16',
    # dtype='float16',  # float16
    optimizer=dict(type=AdamW, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         # # ================= RefSegEncoderDecoder =================
    #         # 'simple_concat_proj':  dict(lr_mult=10.0),
    #         # 'context_gate':        dict(lr_mult=10.0),

    #         # # ================= ReasoningClsHead =================
    #         # 'sam_proj':            dict(lr_mult=10.0),
    #         # 'fusion_conv':         dict(lr_mult=10.0),
    #         # 'proto_adapter':       dict(lr_mult=10.0),
    #         # 'function_prototypes': dict(lr_mult=10.0),
    #         # 'reasoning_engine':    dict(lr_mult=10.0),
    #         # 'feat_to_proto_proj':  dict(lr_mult=10.0),
    #         # 'consistency_head':    dict(lr_mult=10.0),
    #     }
    # ),
    # accumulative_counts=4, # 等效 batch size 调整
)

