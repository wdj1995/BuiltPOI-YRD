from tqdm import tqdm
import torch

from utils import compute_metrics


def run_one_epoch(model, dataloader, criterion, device, optimizer=None) -> dict:
    """执行一个 epoch；传入 optimizer 时为训练，否则为验证/测试。"""
    # 通过 optimizer 是否为空判断当前阶段，避免训练和评估写两套重复循环。
    is_train = optimizer is not None
    model.train(is_train)

    # total_loss 按样本数累计，最后除以数据集大小得到平均 loss。
    total_loss = 0.0
    # 保存所有 batch 的真实标签和预测标签，用于 epoch 结束后统一计算 OA 和 macro F1。
    y_true = []
    y_pred = []

    # 训练阶段需要梯度；验证/测试阶段关闭梯度以减少显存和计算开销。
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(dataloader, leave=False):
            # Dataset 返回的三个核心字段：影像、POI 文本特征、建筑功能标签。
            image        = batch["image"].to(device)
            text_feature = batch["text_feature"].to(device)
            label        = batch["fun_cls"].to(device)

            # 前向传播：ResNet 图像特征和 Word2Vec 文本特征在模型内部融合。
            logits = model(image, text_feature)
            loss = criterion(logits, label)

            if is_train:
                # 标准反向传播流程：清梯度、反传、更新参数。
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # loss.item() 是当前 batch 的平均 loss，乘以 batch size 后累计为样本级总 loss。
            total_loss += loss.item() * label.size(0)
            # logits 最大值所在位置就是预测类别。
            pred = logits.argmax(dim=1)
            y_true.extend(label.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    # 统一计算分类指标：OA 即总体准确率，macro_f1 为三类 F1 的宏平均。
    metrics         = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics
