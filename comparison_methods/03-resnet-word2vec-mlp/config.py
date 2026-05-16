from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data split files. Each jsonl row should contain image_id, poi_prompt, and fun_cls.
    train_jsonl  : str = "./data/splits/train.jsonl"
    val_jsonl    : str = "./data/splits/val.jsonl"
    test_jsonl   : str = "./data/splits/test.jsonl"

    # Image and mask directories. image_id is combined with image_ext/mask_ext when needed.
    image_root   : str = "F:/Python_Files/Python_Project_02/Text_Image_Learning/RSRefSeg-function-seg-1202/dataset/JL_Images"
    mask_root    : str = "F:/Python_Files/Python_Project_02/Text_Image_Learning/RSRefSeg-function-seg-1202/dataset/building_labels"
    image_ext    : str = ".tif"
    mask_ext     : str = ".png"

    # Local pretrained model paths.
    word2vec_path     : str = "./download_models/word2vec-google-news-300.model"
    resnet_model_dir  : str = "./download_models/resnet-50"

    # Precomputed Word2Vec feature root. It should contain train/val/test subfolders.
    text_feature_root : str = "./process_data/word2vec_features"

    # Output directory for checkpoints and TensorBoard logs.
    output_dir   : str = "outputs"

    # Task and image input settings.
    num_classes  : int = 3
    image_size   : int = 224
    bbox_expand_ratio : float = 0.3

    # Training hyperparameters.
    batch_size   : int = 16
    num_workers  : int = 4
    epochs       : int = 100
    lr           : float = 1e-4
    resnet_lr    : float = 1e-5
    weight_decay : float = 1e-4
    seed         : int = 42

    # Early stopping: monitor validation macro F1, larger is better.
    early_stopping_patience  : int = 10
    early_stopping_min_delta : float = 1e-4

    # Model dimensions.
    image_feature_dim : int = 512
    text_feature_dim  : int = 256
    fusion_dim        : int = 512
    dropout           : float = 0.3

    # Freeze all ResNet-50 stages first, then unfreeze the last N stages.
    freeze_resnet           : bool = True
    resnet_trainable_stages : int = 1
    device                  : str = "cuda"

    def ensure_output_dir(self) -> Path:
        """Create output directory if needed and return it as a Path."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
