import argparse

import torch
import torch.nn as nn

from config import Config
from dataset import build_dataloader
from engine import run_one_epoch
from model import MultiModalResNetBert
from utils import get_device, set_seed


def evaluate_model(split: str = "test", checkpoint_path: str = "outputs/best_model.pth") -> dict:
    """加载指定 checkpoint，并在 val/test 数据集上计算 loss、OA 和 macro F1。"""
    config = Config()
    set_seed(config.seed)
    device = get_device(config.device)

    # 评估阶段不打乱数据，保证输出结果稳定且便于复查样本顺序。
    dataloader = build_dataloader(config, split, shuffle=False)

    # 评估时必须使用与训练阶段一致的模型结构和冻结配置。
    model = MultiModalResNetBert(
        resnet_model_dir=config.resnet_model_dir,
        bert_model_dir=config.bert_model_dir,
        num_classes=config.num_classes,
        image_feature_dim=config.image_feature_dim,
        text_feature_dim=config.text_feature_dim,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
        freeze_resnet=config.freeze_resnet,
        resnet_trainable_stages=config.resnet_trainable_stages,
        bert_frozen_layers=config.bert_frozen_layers,
        freeze_bert_embeddings=config.freeze_bert_embeddings,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = run_one_epoch(model, dataloader, nn.CrossEntropyLoss(), device)
    print(
        f"{split} loss {metrics['loss']:.4f} | "
        f"OA {metrics['OA']:.4f} | macro_f1 {metrics['macro_f1']:.4f}"
    )
    return metrics


def main() -> None:
    """命令行评估入口，支持选择 val/test split 和 checkpoint 路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    args = parser.parse_args()

    evaluate_model(split=args.split, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
