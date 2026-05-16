import json
from pathlib import Path

import numpy as np
import torch
from gensim.models import KeyedVectors
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocess import sentence_to_vector


class BuildingFunctionDataset(Dataset):
    """建筑功能分类数据集：同步读取遥感影像、建筑 mask、POI 文本特征和类别标签。"""

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        mask_root: str,
        word2vec_path: str,
        split: str,
        text_feature_root: str = None,
        image_ext: str = ".tif",
        mask_ext: str = ".png",
        image_size: int = 224,
        bbox_expand_ratio: float = 0.3,
    ) -> None:
        # jsonl 中保存样本索引信息，包括 image_id、poi_prompt 和 fun_cls。
        self.records = self._load_jsonl(jsonl_path)

        # image_id 会分别与 image_root/mask_root 组合，定位 tif 影像和 png mask。
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)

        # 如果提前运行 preprocess.py，这里会指向对应 split 的 .npy 文本特征目录。
        self.text_feature_root = Path(text_feature_root) / split if text_feature_root else None
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.bbox_expand_ratio = bbox_expand_ratio

        # 默认优先读取预计算文本特征；如果没有特征目录，则退回到现场加载 Word2Vec 计算。
        self.word2vec_path = word2vec_path
        self.w2v = None
        self.embedding_dim = 300
        if self.text_feature_root is None or not self.text_feature_root.exists():
            # 这里只临时加载一次以确认向量维度，不把大模型常驻在主进程里。
            keyed_vectors = KeyedVectors.load(word2vec_path, mmap="r")
            self.embedding_dim = keyed_vectors.vector_size
            del keyed_vectors
        else:
            # 从已经保存的 .npy 文件中推断文本特征维度，通常是 300。
            self.embedding_dim = self._infer_feature_dim()

        # ResNet 使用 ImageNet 预训练权重，因此采用对应的均值和方差归一化。
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_jsonl(jsonl_path: str) -> list[dict]:
        """逐行读取 jsonl，并检查每个样本是否包含必要字段。"""
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

    def _infer_feature_dim(self) -> int:
        """读取第一个样本的 .npy 特征，用于确定文本特征维度。"""
        first_id = str(self.records[0]["image_id"])
        feature_path = self._feature_path(first_id)
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Text feature not found: {feature_path}. "
                "Run preprocess.py first or remove text_feature_root to compute features on the fly."
            )
        return int(np.load(feature_path).shape[0])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """返回单个训练样本，供 DataLoader 自动拼接成 batch。"""
        record = self.records[idx]
        image_id = str(record["image_id"])

        # image_id 不带后缀时，自动拼接 image_ext/mask_ext。
        image_path = self._build_path(self.image_root, image_id, self.image_ext)
        mask_path = self._build_path(self.mask_root, image_id, self.mask_ext)

        image = Image.open(image_path).convert("RGB")

        # 使用ask外扩生成bbox，基于bbox裁剪遥感影像，并保证bbox不发生变形
        if mask_path.exists():
            image = self._crop_by_square_bbox(image, mask_path, self.bbox_expand_ratio)
        image = self.image_transform(image)

        # 文本侧优先读取预生成的 Word2Vec 句向量，否则现场由 poi_prompt 计算。
        text_feature = self._load_or_encode_text(image_id, record["poi_prompt"])
        label = torch.tensor(int(record["fun_cls"]), dtype=torch.long)

        return {
            "image": image,
            "text_feature": text_feature,
            "fun_cls": label,
        }

    def _load_or_encode_text(self, image_id: str, poi_prompt: str) -> torch.Tensor:
        """读取或计算单个样本的 POI 文本特征。"""
        if self.text_feature_root is not None and self.text_feature_root.exists():
            # 预处理路径存在时，直接读取 preprocess.py 生成的 300 维 .npy 特征。
            feature = np.load(self._feature_path(image_id)).astype(np.float32)
        else:
            # 没有预计算特征时才懒加载本地 Word2Vec 模型，避免每个 epoch 重复大规模初始化。
            if self.w2v is None:
                self.w2v = KeyedVectors.load(self.word2vec_path, mmap="r")
            feature = sentence_to_vector(poi_prompt, self.w2v).astype(np.float32)
        return torch.tensor(feature, dtype=torch.float32)

    def _feature_path(self, image_id: str) -> Path:
        """根据 image_id 定位对应的文本特征文件。"""
        return self.text_feature_root / f"{Path(image_id).stem}.npy"

    @staticmethod
    def _build_path(root: Path, image_id: str, ext: str) -> Path:
        """根据 image_id 构造影像或 mask 文件路径。"""
        path = Path(image_id)
        if path.suffix:
            return root / path
        return root / f"{image_id}{ext}"

    @staticmethod
    def _crop_by_square_bbox(image: Image.Image, mask_path: Path, expand_ratio: float) -> Image.Image:
        """
        根据建筑 mask 生成扩展后的正方形 bbox，并裁剪原始 RGB 影像。

        注意：
            1. mask 只用于定位建筑区域；
            2. 不再将 mask 外区域置黑；
            3. bbox 会被调整为正方形，避免 resize 到 224×224 时发生形变；
            4. expand_ratio 用于保留一定建筑周边上下文。
        """
        mask       = Image.open(mask_path).convert("L").resize(image.size)

        mask_array = np.array(mask) > 0
        ys, xs     = np.where(mask_array)

        if len(xs) == 0 or len(ys) == 0:
            return image

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1

        bbox_width  = x_max - x_min
        bbox_height = y_max - y_min

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        # 以较长边作为正方形边长，避免建筑形状被拉伸。
        square_size = max(bbox_width, bbox_height)

        # 在正方形边长基础上继续扩展，用于保留周边道路、绿地、邻近建筑等上下文。
        square_size = square_size * (1.0 + 2.0 * expand_ratio)

        half_size   = square_size / 2.0

        left  = int(round(center_x - half_size))
        upper = int(round(center_y - half_size))
        right = int(round(center_x + half_size))
        lower = int(round(center_y + half_size))

        # 防止 crop box 超出原始影像边界。
        left  = max(0, left)
        upper = max(0, upper)
        right = min(image.width, right)
        lower = min(image.height, lower)

        # 极端情况下，如果裁剪框无效，则退回原图。
        if right <= left or lower <= upper:
            return image

        return image.crop((left, upper, right, lower))


def build_dataloader(config, split: str, shuffle: bool) -> DataLoader:
    """根据配置和 split 名称创建 PyTorch DataLoader。"""
    jsonl_map = {
        "train": config.train_jsonl,
        "val": config.val_jsonl,
        "test": config.test_jsonl,
    }

    dataset = BuildingFunctionDataset(
        jsonl_path=jsonl_map[split],
        image_root=config.image_root,
        mask_root=config.mask_root,
        word2vec_path=config.word2vec_path,
        split=split,
        text_feature_root=config.text_feature_root,
        image_ext=config.image_ext,
        mask_ext=config.mask_ext,
        image_size=config.image_size,
        bbox_expand_ratio=config.bbox_expand_ratio,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
    )
