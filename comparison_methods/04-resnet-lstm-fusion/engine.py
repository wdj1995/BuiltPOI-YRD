from tqdm import tqdm
import torch

from utils import compute_metrics


def run_one_epoch(model, dataloader, criterion, device, optimizer=None) -> dict:
    """执行一个 epoch；传入 optimizer 时为训练，否则为验证或测试。"""
    is_train = optimizer is not None
    model.train(is_train)

    # loss 按样本数累计，最后除以数据集大小得到平均 loss。
    total_loss = 0.0
    y_true = []
    y_pred = []

    # 验证和测试阶段关闭梯度，减少显存和计算开销。
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(dataloader, leave=False):
            image = batch["image"].to(device)
            text_feature = batch["text_feature"].to(device)
            text_length = batch["text_length"].to(device)
            label = batch["fun_cls"].to(device)

            # 模型内部完成 ResNet 视觉编码、BiLSTM 文本编码和多模态融合分类。
            logits = model(image, text_feature, text_length)
            loss = criterion(logits, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * label.size(0)
            pred = logits.argmax(dim=1)
            y_true.extend(label.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics
