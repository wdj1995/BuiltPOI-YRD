import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证不同运行之间的实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关闭 cudnn benchmark，以换取更稳定的可复现行为。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """根据配置选择计算设备；CUDA 不可用时自动退回 CPU。"""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred) -> dict:
    """计算分类评估指标：OA 和 macro F1。"""
    return {
        # OA: Overall Accuracy，即所有样本上的总体分类准确率。
        "OA": accuracy_score(y_true, y_pred),
        # macro_f1: 对每个类别分别计算 F1 后取平均，更适合类别不均衡场景。
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def save_checkpoint(state: dict, output_dir: str, filename: str = "best_model.pth") -> None:
    """保存模型 checkpoint，包括模型参数、优化器状态和当前最佳指标等信息。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(output_dir) / filename)
