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
# Description: "The model structure has been reconstructed, integrating geographical text information and location information."


import copy
import logging
import random
from typing import Dict, List, Union, Mapping
import datasets
import mmcv
from mmengine import Config, list_from_file, print_log
from mmengine.dataset import Compose, BaseDataset
from torch.utils.data import Dataset
from mmseg.datasets import LoadAnnotations
from mmseg.registry import DATASETS, TRANSFORMS
import os
from PIL import Image
from mmcv.transforms.processing import Resize
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.structures import LabelData


@DATASETS.register_module()
class RefSegDataset(Dataset):
    METAINFO = dict(
        function_classes=['Residential', 'Commercial-Industrial', 'Public service']
    )
    def __init__(self,
                 data_root: str,
                 building_labels_root: str,
                 ann_file: str,  # jsonl file
                 pipeline: List[Dict] = None,
                 test_mode: bool = False,
                 reduce_zero_label: bool = False,
                 metainfo: Union[Mapping, Config, None] = None,
                 ):
        self.data_root = data_root
        self.building_labels_root = building_labels_root
        self.ann_file = ann_file
        self.dataset = datasets.load_dataset('json', data_files=ann_file)['train']  # always use the train split if loaded from jsonl
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.reduce_zero_label = reduce_zero_label
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))


    @classmethod
    def _load_metainfo(cls, metainfo: Union[Mapping, Config, None] = None) -> dict:
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo


    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def __len__(self):
        return len(self.dataset)


    # 支持 ClassAwareSampler 的核心方法
    def get_cat_ids(self, idx: int) -> List[int]:
        """获取单个样本的类别 ID。返回必须是一个列表（MMEngine 规范）。"""
        item = self.dataset[idx]
        # 直接从底层的 jsonl 数据中读取分类标签
        label = int(item['fun_cls'])
        return [label]
    # 如果你希望采样器运行得飞快，可以预先提取所有标签（可选优化）
    def get_labels(self):
        """一次性返回所有标签，有些版本的 Sampler 可能会用到"""
        return [int(item['fun_cls']) for item in self.dataset]


    def get_item(self, idx: int):
        item = self.dataset[idx]

        results = dict()
        results['img_path'] = os.path.join(self.data_root, item['file_name'])
        results['building_seg'] = os.path.join(self.building_labels_root, item['image_id'] + ".png")  # 拼接masks文件路径
        results['gt_label'] = int(item['fun_cls']) # 建筑功能标注，添加到json标注文件中
        results['poi_text'] = item['poi_text']
        results['poi_prompt'] = item['poi_prompt']

        # 原始文本示例: "Urban building context: the immediate...; the nearby...; the outer..."
        raw_prompt = item['poi_prompt']
        # 去除前缀 (如果有的话)
        clean_prompt = raw_prompt.replace("Urban building context:", "").strip()
        # 物理分割：按分号拆解
        parts = clean_prompt.split(';')
        # 赋值给三个独立字段
        # Level 1: Immediate surroundings (0-100m) -> poi_prompt1
        results['poi_prompt1'] = parts[0].strip()
        # Level 2: Nearby area (100-200m) -> poi_prompt2
        results['poi_prompt2'] = parts[1].strip()
        # Level 3: Outer area (200-300m) -> poi_prompt3
        results['poi_prompt3'] = parts[2].strip()

        results['image_id'] = item['image_id']
        results['reduce_zero_label'] = self.reduce_zero_label
        results['seg_fields'] = []

        results = self.pipeline(results)
        return results


    def __getitem__(self, idx: int):
        try:
            return self.get_item(idx)
        except Exception as e:
            print('Error in RefSegDataset.__getitem__:', e)
            return self.get_item(random.randint(0, len(self.dataset)))



@TRANSFORMS.register_module()
class LoadSegAnnotations(LoadAnnotations):
    def _load_seg_map(self, results: dict) -> None:

        gt_building_seg = np.array(Image.open(results['building_seg']))

        # # reduce zero_label
        # if self.reduce_zero_label is None:
        #     self.reduce_zero_label = results['reduce_zero_label']
        # assert self.reduce_zero_label == results['reduce_zero_label'], \
        #     'Initialize dataset with `reduce_zero_label` as ' \
        #     f'{results["reduce_zero_label"]} but when load annotation ' \
        #     f'the `reduce_zero_label` is {self.reduce_zero_label}'
        #
        # if self.reduce_zero_label:
        #     # avoid using underflow conversion
        #     gt_function_seg[gt_function_seg == 0] = 255
        #     gt_function_seg = gt_function_seg - 1
        #     gt_function_seg[gt_function_seg == 254] = 255

        img_h, img_w = results['ori_shape'][:2]
        mask = gt_building_seg
        mask_h, mask_w = mask.shape
        if (mask_h, mask_w) != (img_h, img_w):
            mask = mmcv.imresize(mask, (img_w, img_h), interpolation='nearest')

        results['gt_building_seg'] = mask
        results['seg_fields'].append('gt_building_seg')


@TRANSFORMS.register_module()
class NewResize(Resize):
    """继承原 Resize, 支持多个 seg mask 缩放"""
    def __init__(self,
                 *args,
                 resize_labels: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_labels = resize_labels# 存储新参数

    def _resize_seg(self, results: dict) -> None:
        """Resize multiple semantic segmentation maps."""
        # 如果配置为 False (例如在 test_pipeline 中)，
        if not self.resize_labels:
            # 则*不执行*任何操作，直接返回。
            # 标签保持原始尺寸 (500x500)。
            return

        # 如果resize_labels=True` (即 train_pipeline) 时执行
        # 遍历 seg_fields 中登记的 mask
        for key in ['gt_seg_map', 'gt_building_seg']:
            if key in results and results[key] is not None:
                if not isinstance(results[key], np.ndarray):
                    results[key] = np.array(results[key])
                if self.keep_ratio:
                    seg = mmcv.imrescale(
                        results[key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    seg = mmcv.imresize(
                        results[key],
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[key] = seg


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """将字典中指定键的 NumPy 数组转换为 PyTorch Tensor。"""
    def __init__(self, keys):
        """
        Args:
            keys (list[str]): 需要转换为 Tensor 的键列表。
        """
        self.keys = keys
    def transform(self, results: dict) -> dict:
        """将指定键的值从 NumPy 数组转换为 Tensor。"""
        for key in self.keys:
            if key in results and isinstance(results[key], np.ndarray):
                numpy_array = results[key]
                # 检查数组是否为 2D (H, W)
                if numpy_array.ndim == 2:
                    # 如果是 2D，在 axis=0 处添加一个新维度，变为 (1, H, W)
                    numpy_array = np.expand_dims(numpy_array, axis=0)
                # 将 (1, H, W) 或 (C, H, W) 数组转换为 Tensor
                results[key] = torch.from_numpy(numpy_array).long()
        return results


@TRANSFORMS.register_module()
class PackMultiTaskInputs(BaseTransform):
    """
    自定义打包类，用于处理 [建筑分割] 和 [功能分类] 的标签。
    功能：
        1. 将图像转换为 Tensor 并处理为 (C, H, W)。
        2. 将 gt_building_seg 封装为 PixelData 并放入 data_sample。
        3. 收集元数据 (metainfo)。
    """
    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'poi_text', 'poi_prompt')):
        self.meta_keys = meta_keys

    def _get_bbox_from_mask(self, mask_tensor):
        """
            从二值 Mask Tensor 计算外接矩形 BBox。
        """
        # 确保是 2D
        if mask_tensor.ndim == 3:
            mask_2d = mask_tensor.squeeze(0)
        else:
            mask_2d = mask_tensor

        # 获取非零元素坐标
        # mask_2d > 0 返回 bool mask, .nonzero() 返回坐标 indices [N, 2] (y, x)
        nonzero_indices = torch.nonzero(mask_2d > 0)

        h, w = mask_2d.shape

        if nonzero_indices.shape[0] == 0:
            # 如果 mask 全黑（无建筑），返回全图 bbox 或 [0,0,0,0]
            # 这里为了 SAM 稳定性，通常返回全图或者一个极小框，这里返回全图作为 Context
            return torch.tensor([[0.0, 0.0, float(w), float(h)]], dtype=torch.float32)

        y_min = torch.min(nonzero_indices[:, 0]).float()
        y_max = torch.max(nonzero_indices[:, 0]).float()
        x_min = torch.min(nonzero_indices[:, 1]).float()
        x_max = torch.max(nonzero_indices[:, 1]).float()

        # 结果格式 [x1, y1, x2, y2]
        # +1 是为了包含边界像素 (可选，视具体需求，通常 box 是左闭右开或闭闭，这里采用包围框)
        return torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        # --- 1. 处理图像 (Inputs) ---
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # 调整为 (C, H, W) 并转 Tensor
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1)) # 无论原本是否连续，Transpose 后必须强制变为连续数组，再转 Tensor
                img = to_tensor(img) # 注意：此时 img 已经是 (C, H, W) 且内存连续，无需再调 .contiguous()
            packed_results['inputs'] = img

        # --- 2. 初始化 DataSample ---
        data_sample = SegDataSample()
        # 处理 Graph Data
        if 'graph_data' in results:
            data_sample.graph_data = results['graph_data']

        # --- 3. 处理 建筑分割 GT (gt_building_seg) ---
        if 'gt_building_seg' in results:
            img_gt = results['gt_building_seg']
            # 转 Tensor
            if isinstance(img_gt, np.ndarray):
                if img_gt.ndim == 2:
                    img_gt = img_gt[None, ...]
                data_building = to_tensor(img_gt.astype(np.int64))
            elif isinstance(img_gt, torch.Tensor):
                data_building = img_gt
            else:
                raise TypeError(f"Type of gt_building_seg must be numpy or tensor, but got {type(img_gt)}")

            # A. 封装 Mask 进 PixelData
            data_sample.gt_building_seg = PixelData(data=data_building)

            # B. 计算 BBox 并封装进 InstanceData
            # 注意：data_building 此时 shape 为 [1, H, W]
            bbox = self._get_bbox_from_mask(data_building)

            gt_instances = InstanceData()
            gt_instances.bboxes = bbox  # Tensor [1, 4]
            # SAM 需要 labels 吗？通常 instance分割需要，这里给个默认 label 1
            gt_instances.labels = torch.tensor([1], dtype=torch.long)
            # 赋值给 data_sample
            data_sample.gt_instances = gt_instances

        # --- 4. 处理 gt_label (分类任务) ---
        # 以前是 gt_function_seg (PixelData)，现在是 gt_label (LabelData)
        if 'gt_label' in results:
            # 确保它是 Tensor, shape 为 [1] 或者 scalar
            label_val = results['gt_label']
            if isinstance(label_val, int):
                label_tensor = torch.tensor([label_val], dtype=torch.long)
            elif isinstance(label_val, torch.Tensor):
                label_tensor = label_val.long()
            else:
                label_tensor = torch.tensor([int(label_val)], dtype=torch.long)

            # SegDataSample 没有 set_gt_label 方法，需要手动封装 LabelData
            gt_label_data = LabelData()
            gt_label_data.label = label_tensor
            # 动态给 data_sample 绑定一个名为 gt_label 的属性
            data_sample.gt_label = gt_label_data

        # --- 5. 处理元数据 (Meta Info) ---
        # 只存放文件名、POI文本等不需要上 GPU 的信息
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results



if __name__ == '__main__':
    dataset_root = r'.../datainfo'
    ann_file = 'val.jsonl'
    # tokenizer = 'lcybuaa1111/Git-RSCLIP'
    pipeline = [
        # dict(type='mmseg.LoadImageFromFile'),
        dict(type='mmseg.LoadSegAnnotations'),
        # dict(type='mmseg.DefaultFormatBundle'),
        # dict(type='mmseg.PackSegInputs', keys=['gt_seg_map'], meta_keys=['filename']),
    ]
    dataset = RefSegDataset(dataset_root, ann_file, pipeline)
    print(len(dataset))
    print(dataset[0])

