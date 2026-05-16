import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset import build_dataloader
from model import MultiModalResNetBert
from utils import get_device, save_checkpoint, set_seed


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """统计模型总参数量和可训练参数量，用于确认冻结配置是否生效。"""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def build_optimizer(model: MultiModalResNetBert, config: Config) -> torch.optim.Optimizer:
    """为 ResNet、BERT 和下游融合分类层设置不同学习率。"""
    resnet_params = []
    bert_params = []
    downstream_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 预训练 backbone 使用较小学习率，新建投影层和分类头使用普通学习率。
        if name.startswith("image_encoder.backbone"):
            resnet_params.append(param)
        elif name.startswith("text_encoder.backbone"):
            bert_params.append(param)
        else:
            downstream_params.append(param)

    param_groups = []
    if resnet_params:
        param_groups.append({"params": resnet_params, "lr": config.resnet_lr, "name": "resnet"})
    if bert_params:
        param_groups.append({"params": bert_params, "lr": config.bert_lr, "name": "bert"})
    if downstream_params:
        param_groups.append({"params": downstream_params, "lr": config.lr, "name": "downstream"})

    print(f"Trainable ResNet parameters: {sum(p.numel() for p in resnet_params):,}")
    print(f"Trainable BERT parameters: {sum(p.numel() for p in bert_params):,}")
    print(f"Trainable downstream parameters: {sum(p.numel() for p in downstream_params):,}")
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def get_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float | None:
    """根据参数组名称读取当前学习率，便于写入 TensorBoard。"""
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return None


def write_tensorboard_scalars(
    writer: SummaryWriter,
    train_metrics: dict,
    val_metrics: dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """将每个 epoch 的指标和各参数组学习率写入 TensorBoard。"""
    writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
    writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
    writer.add_scalar("OA/train", train_metrics["OA"], epoch)
    writer.add_scalar("OA/val", val_metrics["OA"], epoch)
    writer.add_scalar("Macro_F1/train", train_metrics["macro_f1"], epoch)
    writer.add_scalar("Macro_F1/val", val_metrics["macro_f1"], epoch)

    for group_name in ("downstream", "resnet", "bert"):
        lr = get_group_lr(optimizer, group_name)
        if lr is not None:
            writer.add_scalar(f"Learning_Rate/{group_name}", lr, epoch)


def main() -> None:
    """训练入口：构建数据、初始化模型、训练验证、保存最优 checkpoint。"""
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
    print(f"BERT model directory: {config.bert_model_dir}")
    print(f"Freeze ResNet-50 first: {config.freeze_resnet}")
    print(f"Unfreeze last ResNet stage count: {config.resnet_trainable_stages}")
    print(f"Freeze BERT embeddings: {config.freeze_bert_embeddings}")
    print(f"Frozen BERT encoder layer count: {config.bert_frozen_layers}")
    print(f"Downstream lr: {config.lr}")
    print(f"ResNet lr: {config.resnet_lr}")
    print(f"BERT lr: {config.bert_lr}")
    print(
        "Early stopping: "
        f"monitor=val macro_f1, patience={config.early_stopping_patience}, "
        f"min_delta={config.early_stopping_min_delta}"
    )
    print("=" * 80)

    print("Building train/val dataloaders...")
    # DataLoader 会返回 image、input_ids、attention_mask 和 fun_cls。
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max BERT text length: {config.max_text_length}")

    print("Initializing model...")
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
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    # 延迟导入避免简单查看配置或构建模型时额外加载训练循环依赖。
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

            # 同时记录 train/val 指标和不同模块的学习率，方便后续复盘训练曲线。
            write_tensorboard_scalars(
                writer=writer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                optimizer=optimizer,
                epoch=epoch,
            )

            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train loss {train_metrics['loss']:.4f} OA {train_metrics['OA']:.4f} macro_f1 {train_metrics['macro_f1']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} OA {val_metrics['OA']:.4f} macro_f1 {val_metrics['macro_f1']:.4f}"
            )

            # 以验证集 macro F1 作为保存最优模型的依据，更适合类别不均衡场景。
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
