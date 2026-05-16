from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # 数据划分文件：每行是一个 JSON 对象，至少包含 image_id、poi_prompt 和 fun_cls。
    train_jsonl: str = "./data/splits/train.jsonl"
    val_jsonl  : str = "./data/splits/val.jsonl"
    test_jsonl : str = "./data/splits/test.jsonl"

    # 遥感影像和建筑 mask 路径；image_id 不带后缀时会自动拼接 image_ext/mask_ext。
    image_root : str = "F:/Python_Files/Python_Project_02/Text_Image_Learning/RSRefSeg-function-seg-1202/dataset/JL_Images"
    mask_root  : str = "F:/Python_Files/Python_Project_02/Text_Image_Learning/RSRefSeg-function-seg-1202/dataset/building_labels"
    image_ext  : str = ".tif"
    mask_ext   : str = ".png"

    # 本地 HuggingFace 格式预训练模型目录，代码使用 local_files_only=True，不会联网下载。
    resnet_model_dir : str = "./download_models/resnet-50"
    bert_model_dir   : str = "./download_models/bert-base-uncased"

    # checkpoint、TensorBoard 日志等训练输出目录。
    output_dir       : str = "outputs"

    # 分类任务和图像输入设置。
    num_classes : int = 3
    image_size  : int = 224
    # 根据建筑 mask 裁剪 bbox 时向外扩展的比例，用于保留周边上下文。
    bbox_expand_ratio: float = 0.3

    # 通用训练超参数。
    batch_size  : int = 16
    num_workers : int = 0
    epochs      : int = 100
    # downstream lr 用于新建投影层和融合分类头。
    lr          : float = 1e-4
    # ResNet 与 BERT backbone 通常使用更小学习率微调。
    resnet_lr   : float = 1e-5
    bert_lr     : float = 1e-5
    weight_decay: float = 1e-4
    seed        : int = 42

    # 早停策略：验证集 macro F1 连续若干 epoch 无明显提升时停止训练。
    early_stopping_patience  : int = 10
    early_stopping_min_delta : float = 1e-4

    # 多模态特征维度：图像与文本先投影到固定维度，再拼接进入 MLP。
    image_feature_dim : int = 512
    text_feature_dim  : int = 256
    fusion_dim        : int = 512
    dropout           : float = 0.3

    # BERT 文本分支设置。
    max_text_length        : int = 64
    # 是否冻结 BERT embedding 层。
    freeze_bert_embeddings : bool = False
    # 冻结 BERT encoder 的前 N 层；BERT-base 一般共有 12 层。
    bert_frozen_layers     : int = 0

    # ResNet-50 微调设置：先冻结整个 backbone，再按需解冻最后 N 个 stage。
    freeze_resnet          : bool = True
    resnet_trainable_stages: int = 1
    device                 : str = "cuda"

    def ensure_output_dir(self) -> Path:
        """确保输出目录存在，并返回 Path 对象。"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
