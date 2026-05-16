import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class BuildingFunctionDataset(Dataset):
    """建筑功能分类数据集：同步读取影像、mask、POI 文本和类别标签。"""
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        mask_root: str,
        bert_model_dir: str,
        split: str,
        image_ext: str = ".tif",
        mask_ext: str = ".png",
        image_size: int = 224,
        max_text_length: int = 64,
        bbox_expand_ratio: float = 0.3,
    ) -> None:
        # jsonl 保存样本索引信息，包括 image_id、poi_prompt 和 fun_cls。
        self.records = self._load_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.max_text_length = max_text_length
        self.bbox_expand_ratio = bbox_expand_ratio
        self.split = split

        # tokenizer 只负责把文本转为 BERT 输入，不在 Dataset 中加载 BERT 编码器。
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_dir, local_files_only=True)

        # ResNet 使用 ImageNet 预训练权重，因此沿用 ImageNet mean/std 归一化。
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_jsonl(jsonl_path: str) -> list[dict]:
        """逐行读取 jsonl，并检查训练所需字段是否齐全。"""
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {jsonl_path}:{line_no}") from exc

                required = {"image_id", "poi_prompt", "fun_cls"}
                missing = required - set(record)
                if missing:
                    raise ValueError(f"{jsonl_path}:{line_no} missing fields: {sorted(missing)}")
                records.append(record)

        if not records:
            raise ValueError(f"No samples found in {jsonl_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """返回单个样本，DataLoader 会自动拼接为 batch。"""
        record = self.records[idx]
        image_id = str(record["image_id"])

        # image_id 不带后缀时，自动拼接配置中的 image_ext/mask_ext。
        image_path = self._build_path(self.image_root, image_id, self.image_ext)
        mask_path = self._build_path(self.mask_root, image_id, self.mask_ext)

        image = Image.open(image_path).convert("RGB")
        if mask_path.exists():
            image = self._crop_by_square_bbox(image, mask_path, self.bbox_expand_ratio)
        image = self.image_transform(image)

        encoded_text = self._encode_text(record["poi_prompt"])
        label = torch.tensor(int(record["fun_cls"]), dtype=torch.long)

        return {
            "image": image,
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
            "fun_cls": label,
        }

    def _encode_text(self, poi_prompt: str) -> dict[str, torch.Tensor]:
        """将 POI 文本编码为固定长度 BERT token 序列。"""
        encoded = self.tokenizer(
            poi_prompt,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0).long(),
            "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        }

    @staticmethod
    def _build_path(root: Path, image_id: str, ext: str) -> Path:
        """根据 image_id 构造影像或 mask 路径。"""
        path = Path(image_id)
        if path.suffix:
            return root / path
        return root / f"{image_id}{ext}"

    @staticmethod
    def _crop_by_square_bbox(image: Image.Image, mask_path: Path, expand_ratio: float) -> Image.Image:
        """
            根据建筑 mask 生成扩展后的正方形 bbox，并裁剪原始 RGB 影像。

            mask 只用于定位建筑区域；裁剪框会保留一定上下文，并尽量避免
            resize 到 224x224 时产生明显形变。
        """
        mask = Image.open(mask_path).convert("L").resize(image.size)

        mask_array = np.array(mask) > 0
        ys, xs = np.where(mask_array)

        if len(xs) == 0 or len(ys) == 0:
            return image

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        # 以较长边作为正方形边长，避免建筑形状在 resize 时被拉伸。
        square_size = max(bbox_width, bbox_height)
        # 在正方形边长基础上继续扩展，用于保留道路、绿地、邻近建筑等上下文。
        square_size = square_size * (1.0 + 2.0 * expand_ratio)
        half_size = square_size / 2.0

        left = int(round(center_x - half_size))
        upper = int(round(center_y - half_size))
        right = int(round(center_x + half_size))
        lower = int(round(center_y + half_size))

        # 防止 crop box 超出原始影像边界。
        left = max(0, left)
        upper = max(0, upper)
        right = min(image.width, right)
        lower = min(image.height, lower)

        if right <= left or lower <= upper:
            return image

        return image.crop((left, upper, right, lower))


def build_dataloader(config, split: str, shuffle: bool) -> DataLoader:
    """根据配置创建指定 split 的 PyTorch DataLoader。"""
    jsonl_map = {
        "train": config.train_jsonl,
        "val": config.val_jsonl,
        "test": config.test_jsonl,
    }

    dataset = BuildingFunctionDataset(
        jsonl_path=jsonl_map[split],
        image_root=config.image_root,
        mask_root=config.mask_root,
        bert_model_dir=config.bert_model_dir,
        split=split,
        image_ext=config.image_ext,
        mask_ext=config.mask_ext,
        image_size=config.image_size,
        max_text_length=config.max_text_length,
        bbox_expand_ratio=config.bbox_expand_ratio,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
    )
