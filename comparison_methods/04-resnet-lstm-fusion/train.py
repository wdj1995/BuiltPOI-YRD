import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset import build_dataloader
from model import MultiModalResNetWord2Vec
from utils import get_device, save_checkpoint, set_seed


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """统计模型总参数量和可训练参数量，用于确认 ResNet-50 解冻范围。"""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def build_optimizer(model: MultiModalResNetWord2Vec, config: Config) -> torch.optim.Optimizer:
    """为解冻的 ResNet 参数和下游融合分类层设置不同学习率。"""
    backbone_params = []
    downstream_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # ResNet backbone 使用较小学习率，其他新建层使用普通学习率。
        if name.startswith("image_encoder.backbone"):
            backbone_params.append(param)
        else:
            downstream_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.resnet_lr})
    if downstream_params:
        param_groups.append({"params": downstream_params, "lr": config.lr})

    print(f"Trainable ResNet parameters: {sum(p.numel() for p in backbone_params):,}")
    print(f"Trainable downstream parameters: {sum(p.numel() for p in downstream_params):,}")
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def write_tensorboard_scalars(
    writer: SummaryWriter,
    train_metrics: dict,
    val_metrics: dict,
    downstream_lr: float,
    resnet_lr: float | None,
    epoch: int,
) -> None:
    """将每个 epoch 的 loss、OA、macro F1 和学习率写入 TensorBoard。"""
    writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
    writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
    writer.add_scalar("OA/train", train_metrics["OA"], epoch)
    writer.add_scalar("OA/val", val_metrics["OA"], epoch)
    writer.add_scalar("Macro_F1/train", train_metrics["macro_f1"], epoch)
    writer.add_scalar("Macro_F1/val", val_metrics["macro_f1"], epoch)
    writer.add_scalar("Learning_Rate/downstream", downstream_lr, epoch)
    if resnet_lr is not None:
        writer.add_scalar("Learning_Rate/resnet", resnet_lr, epoch)


def main() -> None:
    """训练入口：加载数据、初始化模型、训练验证、保存最优模型并执行早停判断。"""
    config = Config()
    set_seed(config.seed)

    output_dir = config.ensure_output_dir()
    device = get_device(config.device)
    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    print("=" * 80)
    print("Start multimodal building function classification training")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard log directory: {tensorboard_dir}")
    print(f"ResNet-50 model directory: {config.resnet_model_dir}")
    print(f"Word2Vec feature root: {config.text_feature_root}")
    print(f"Freeze ResNet-50 first: {config.freeze_resnet}")
    print(f"Unfreeze last ResNet stage count: {config.resnet_trainable_stages}")
    print(f"Downstream lr: {config.lr}")
    print(f"ResNet lr: {config.resnet_lr}")
    print(
        "Early stopping: "
        f"monitor=val macro_f1, patience={config.early_stopping_patience}, "
        f"min_delta={config.early_stopping_min_delta}"
    )
    print("=" * 80)

    print("Building train/val dataloaders...")
    # DataLoader 会返回 image、text_feature、text_length 和 fun_cls。
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    word_dim = train_loader.dataset.embedding_dim
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Word2Vec feature dim: {word_dim}")

    print("Initializing model...")
    # 文本分支输入维度由预生成 Word2Vec 特征自动推断，通常为 300。
    model = MultiModalResNetWord2Vec(
        word_dim=word_dim,
        resnet_model_dir=config.resnet_model_dir,
        num_classes=config.num_classes,
        image_feature_dim=config.image_feature_dim,
        text_feature_dim=config.text_feature_dim,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_num_layers=config.lstm_num_layers,
        lstm_bidirectional=config.lstm_bidirectional,
        freeze_resnet=config.freeze_resnet,
        resnet_trainable_stages=config.resnet_trainable_stages,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    from engine import run_one_epoch

    best_macro_f1 = -1.0
    bad_epochs = 0

    try:
        for epoch in range(1, config.epochs + 1):
            print("-" * 80)
            print(f"Epoch {epoch:03d}/{config.epochs}: training...")
            train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer)

            print(f"Epoch {epoch:03d}/{config.epochs}: validating...")
            val_metrics = run_one_epoch(model, val_loader, criterion, device)

            # 若 ResNet 完全冻结，优化器中可能只有下游参数组。
            resnet_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else None
            downstream_lr = optimizer.param_groups[-1]["lr"]
            write_tensorboard_scalars(
                writer=writer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                downstream_lr=downstream_lr,
                resnet_lr=resnet_lr,
                epoch=epoch,
            )

            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train loss {train_metrics['loss']:.4f} OA {train_metrics['OA']:.4f} macro_f1 {train_metrics['macro_f1']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} OA {val_metrics['OA']:.4f} macro_f1 {val_metrics['macro_f1']:.4f}"
            )

            # 以验证集 macro F1 作为保存 best_model.pth 的依据。
            improved = val_metrics["macro_f1"] > best_macro_f1 + config.early_stopping_min_delta
            if improved:
                best_macro_f1 = val_metrics["macro_f1"]
                bad_epochs = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_macro_f1": best_macro_f1,
                        "config": config.__dict__,
                    },
                    str(output_dir),
                )
                print(f"Saved best model: {output_dir / 'best_model.pth'}")
                print(f"Best val macro_f1 updated to {best_macro_f1:.4f} at epoch {epoch}")
            else:
                bad_epochs += 1
                print(
                    f"No significant val macro_f1 improvement for {bad_epochs}/"
                    f"{config.early_stopping_patience} epoch(s)."
                )

            if bad_epochs >= config.early_stopping_patience:
                print(
                    "Early stopping triggered: "
                    f"val macro_f1 did not improve by at least {config.early_stopping_min_delta} "
                    f"for {config.early_stopping_patience} consecutive epochs."
                )
                break
    finally:
        writer.close()
        print("Training finished.")


if __name__ == "__main__":
    main()
