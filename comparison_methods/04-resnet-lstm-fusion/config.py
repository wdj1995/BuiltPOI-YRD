from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # 数据划分文件：每行 json 至少包含 image_id、poi_prompt 和 fun_cls。
    train_jsonl  : str = "./data/splits/train.jsonl"
    val_jsonl    : str = "./data/splits/val.jsonl"
    test_jsonl   : str = "./data/splits/test.jsonl"

    # 遥感影像和建筑 mask 目录；image_id 不带后缀时会自动拼接 image_ext/mask_ext。
    image_root   : str = "F:/Python_Files/Python_Project_02/Text_Image_Learning/RSRefSeg-function-seg-1202/dataset/JL_Images"
    mask_root    : str = "F:/Python_Files/Python_Project_02/Text_Image_Learning/RSRefSeg-function-seg-1202/dataset/building_labels"
    image_ext    : str = ".tif"
    mask_ext     : str = ".png"

    # 本地预训练模型路径：训练和预处理均从本地读取，不联网下载。
    word2vec_path     : str = "./download_models/word2vec-google-news-300/word2vec-google-news-300.model"
    resnet_model_dir  : str = "./download_models/resnet-50"

    # 预生成的 Word2Vec 序列特征目录，内部应包含 train/val/test 子目录。
    text_feature_root : str = "./process_data/word2vec_lstm_features"

    # 模型 checkpoint 和 TensorBoard 日志输出目录。
    output_dir   : str = "outputs"

    # 任务和图像输入设置。
    num_classes  : int = 3
    image_size   : int = 224
    bbox_expand_ratio: float = 0.3

    # 训练超参数。
    batch_size   : int = 16
    num_workers  : int = 4
    epochs       : int = 100
    lr           : float = 1e-4
    resnet_lr    : float = 1e-5
    weight_decay : float = 1e-4
    seed         : int = 42

    # 早停策略：监控验证集 macro F1，指标越大越好。
    early_stopping_patience  : int = 10
    early_stopping_min_delta : float = 1e-4

    # 多模态模型维度设置。
    image_feature_dim : int = 512
    text_feature_dim  : int = 256
    fusion_dim        : int = 512
    dropout           : float = 0.3

    # BiLSTM 文本分支设置：POI token 序列过长截断，过短补 0。
    max_tokens        : int = 64
    lstm_hidden_dim   : int = 256
    lstm_num_layers   : int = 1
    lstm_bidirectional: bool = True

    # ResNet-50 微调设置：先冻结 backbone，再按需解冻最后 N 个 stage。
    freeze_resnet           : bool = True
    resnet_trainable_stages : int = 1
    device                  : str = "cuda"

    def ensure_output_dir(self) -> Path:
        """确保输出目录存在，并返回 Path 对象。"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
