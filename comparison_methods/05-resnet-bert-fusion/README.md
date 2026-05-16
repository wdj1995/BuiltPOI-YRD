# ResNet-50 + BERT 多模态建筑功能三分类

本项目用于结合遥感影像中的建筑区域和 POI 文本描述，完成建筑功能三分类任务。

当前代码实现的是 **ResNet-50 视觉分支 + BERT 文本分支 + MLP 融合分类头**。视觉分支提取建筑区域影像特征，文本分支对 `poi_prompt` 进行句子级语义编码，最后将两种模态的特征拼接并输出三分类 logits。

## 代码结构

- `config.py`：集中管理数据路径、模型路径、训练超参数、BERT 冻结配置和输出目录。
- `dataset.py`：读取 jsonl、tif 影像、png mask 和 POI 文本，并构建训练样本。
- `model.py`：定义 ResNet-50 视觉编码器、BERT 文本编码器和多模态融合分类模型。
- `engine.py`：封装一个 epoch 的训练、验证或测试流程，并计算 loss、OA、macro F1。
- `train.py`：完整训练入口，按验证集 `macro_f1` 保存最优模型，并写入 TensorBoard 日志。
- `evaluate.py`：加载 checkpoint，在验证集或测试集上评估。
- `test.py`：默认加载 `outputs/best_model.pth`，在测试集上评估。
- `utils.py`：随机种子、设备选择、指标计算和 checkpoint 保存等工具函数。

## 本地模型文件

项目默认从本地 HuggingFace 格式目录读取预训练模型，不联网下载：

```text
download_models/
├── resnet-50/
└── bert-base-uncased/
```

对应配置位于 `config.py`：

```python
resnet_model_dir = "./download_models/resnet-50"
bert_model_dir = "./download_models/bert-base-uncased"
```

## 模型结构

视觉分支：

```text
cropped image [B, 3, 224, 224]
-> ResNet-50 pooler_output
-> Linear + ReLU + Dropout
-> image feature [B, 512]
```

文本分支：

```text
poi_prompt
-> BERT tokenizer
-> input_ids / attention_mask [B, 64]
-> BERT
-> pooler_output 或 CLS hidden state
-> Linear + ReLU + Dropout
-> text feature [B, 256]
```

融合分类：

```text
concat([image feature, text feature]) -> [B, 768]
-> Linear(768 -> 512)
-> BatchNorm1d
-> ReLU
-> Dropout
-> Linear(512 -> 3)
```

## BERT 冻结配置

可以在 `config.py` 中控制 BERT 的冻结范围：

```python
freeze_bert_embeddings = True
bert_frozen_layers = 10
bert_lr = 1e-5
```

当前配置会冻结 BERT embedding 层和前 10 个 encoder layer。若使用 `bert-base-uncased`，其 encoder 通常共有 12 层，因此默认只有后 2 层 encoder 参与微调。把 `bert_frozen_layers` 改为 `0` 可让所有 encoder 层参与训练；改为 `12` 则冻结全部 encoder 层。

## 数据格式

项目假设 train/val/test 已经提前划分好，并以 jsonl 存放：

```text
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test.jsonl
```

每一行至少包含：

```json
{"image_id": "HZ_000001", "poi_prompt": "Urban building context: ...", "fun_cls": 0}
```

`dataset.py` 会根据 `image_id` 读取对应 tif 影像和 png mask。mask 用于裁剪建筑附近的正方形区域，随后 resize 到 `224x224` 并使用 ImageNet mean/std 归一化。

Dataset 返回的 batch 字段包括：

```text
image
input_ids
attention_mask
fun_cls
```

## 训练与评估

训练：

```bash
python train.py
```

测试：

```bash
python test.py
```

也可以指定评估 split 和 checkpoint：

```bash
python evaluate.py --split test --checkpoint outputs/best_model.pth
```

训练时以验证集 `macro_f1` 作为保存最优模型和早停的依据。
