# ResNet-50 + Word2Vec-BiLSTM 多模态建筑功能三分类

本项目用于结合遥感影像中的建筑区域和 POI 文本描述，完成建筑功能三分类任务。视觉分支使用本地预训练 ResNet-50 提取建筑区域影像特征；文本分支先用本地 Google News Word2Vec 将 `poi_prompt` 编码为定长 token 特征矩阵，再通过 BiLSTM 进行序列语义建模；最后拼接视觉特征和文本特征，并通过 MLP 分类头输出三类 logits。

## 代码结构

- `config.py`：集中管理数据路径、模型路径、训练超参数、BiLSTM 参数和输出目录。
- `preprocess.py`：读取 jsonl 中的 `poi_prompt`，生成 Word2Vec 定长序列特征 `.npz`。
- `dataset.py`：读取 jsonl、tif 影像、png mask、Word2Vec-BiLSTM 文本特征，并构建训练样本。
- `model.py`：定义 ResNet-50 视觉编码器、BiLSTM 文本编码器和多模态融合分类模型。
- `engine.py`：封装一个 epoch 的训练、验证或测试流程，并计算 loss、OA、macro F1。
- `train.py`：完整训练入口，按验证集 `macro_f1` 保存最优模型，并写入 TensorBoard 日志。
- `evaluate.py`：加载 checkpoint，在验证集或测试集上评估。
- `test.py`：默认加载 `outputs/best_model.pth`，在测试集上评估。
- `utils.py`：随机种子、设备选择、指标计算和 checkpoint 保存等工具函数。

## 环境安装

建议先创建独立 Python 环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch`、`torchvision`：模型训练和图像变换。
- `transformers`、`safetensors`：从本地 HuggingFace 格式目录加载 ResNet-50。
- `gensim`：加载本地 Word2Vec `KeyedVectors`。
- `scikit-learn`：计算 OA 和 macro F1。
- `tensorboard`：查看训练和验证曲线。

## 数据格式

项目假设 train/val/test 已经提前划分好，并以 jsonl 存放：

```text
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test.jsonl
```

每一行是一个 JSON 对象，至少包含：

```json
{"image_id": "HZ_000001", "poi_prompt": "Urban building context: ...", "fun_cls": 0}
```

字段说明：

- `image_id`：影像和 mask 的文件名主键；若不带后缀，代码会自动拼接 `.tif` 和 `.png`。
- `poi_prompt`：当前建筑样本对应的 POI 文本描述。
- `fun_cls`：建筑功能类别标签，三分类时取值为 `0, 1, 2`。

影像和 mask 路径在 `config.py` 中配置：

```python
image_root = "你的 tif 影像目录"
mask_root = "你的 png mask 目录"
image_ext = ".tif"
mask_ext = ".png"
```

例如 `image_id = "HZ_000001"` 时，代码会读取：

```text
{image_root}/HZ_000001.tif
{mask_root}/HZ_000001.png
```

## 本地模型文件

项目需要提前准备两个本地预训练模型：

```text
download_models/
├── resnet-50/
│   ├── config.json
│   ├── model.safetensors
│   ├── pytorch_model.bin
│   └── preprocessor_config.json
└── word2vec-google-news-300/
    ├── word2vec-google-news-300.model
    └── word2vec-google-news-300.model.vectors.npy
```

`model.py` 使用：

```python
AutoModel.from_pretrained(config.resnet_model_dir, local_files_only=True)
```

因此 ResNet-50 会从本地目录加载，不会联网下载。

## Word2Vec-BiLSTM 文本特征预处理

训练前建议先生成 POI 文本序列特征：

```bash
python preprocess.py
```

也可以显式指定路径和最大 token 数：

```bash
python preprocess.py --train_jsonl data/splits/train.jsonl --val_jsonl data/splits/val.jsonl --test_jsonl data/splits/test.jsonl --word2vec_path download_models/word2vec-google-news-300/word2vec-google-news-300.model --output_folder process_data/word2vec_lstm_features --max_tokens 64
```

生成结果为每个样本一个 `.npz` 文件：

```text
process_data/word2vec_lstm_features/
├── train/
│   └── HZ_000001.npz
├── val/
│   └── HZ_000101.npz
└── test/
    └── HZ_000201.npz
```

每个 `.npz` 包含：

- `feature`：形状为 `[max_tokens, 300]` 的 Word2Vec token 特征矩阵，默认是 `[64, 300]`。
- `length`：当前样本真实有效 token 数，用于 BiLSTM 的 `pack_padded_sequence`。

生成逻辑：

```text
poi_prompt
-> simple_preprocess 分词
-> 跳过不在 Word2Vec 词表中的 OOV token
-> 超过 max_tokens 的 token 截断
-> 不足 max_tokens 的位置补 0
-> 保存 feature 和 length
```

注意：当前文本分支不再使用 Word2Vec 均值池化句向量，旧的 `[300]` `.npy` 特征不能用于现在的 BiLSTM 模型。

## 图像和 mask 处理

Dataset 的图像处理顺序是：

```text
读取 tif 影像
读取 png mask
将 mask 外的背景像素置为 0
Resize 到 config.image_size，默认 224x224
ToTensor
按 ImageNet mean/std Normalize
输入 ResNet-50
```

因此不需要提前离线把 tif 和 png 缩放到 224。

## 模型结构

视觉分支：

```text
masked image [B, 3, 224, 224]
-> ResNet-50 pooler_output
-> Linear + ReLU + Dropout
-> image feature [B, 512]
```

文本分支：

```text
Word2Vec token sequence [B, 64, 300]
text_length [B]
-> pack_padded_sequence
-> BiLSTM
-> 拼接最后一层前向/后向 hidden state
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

## 训练

确认 `config.py` 中这些路径正确：

```python
train_jsonl = "./data/splits/train.jsonl"
val_jsonl = "./data/splits/val.jsonl"
test_jsonl = "./data/splits/test.jsonl"
image_root = "你的 tif 影像目录"
mask_root = "你的 png mask 目录"
word2vec_path = "./download_models/word2vec-google-news-300/word2vec-google-news-300.model"
resnet_model_dir = "./download_models/resnet-50"
text_feature_root = "./process_data/word2vec_lstm_features"
output_dir = "outputs"
```

先生成文本序列特征：

```bash
python preprocess.py
```

开始训练：

```bash
python train.py
```

训练流程：

```text
加载 train DataLoader
加载 val DataLoader
初始化 ResNet-50 + Word2Vec-BiLSTM 多模态模型
训练一个 epoch
验证一个 epoch
记录 loss、OA、macro F1
如果 val macro_f1 提升，则保存 outputs/best_model.pth
若长期无提升，则触发早停
```

当前早停配置在 `config.py` 中：

```python
early_stopping_patience = 50
early_stopping_min_delta = 1e-4
```

早停监控验证集 `macro_f1`。

## TensorBoard

训练过程会把曲线写入：

```text
outputs/tensorboard
```

查看方式：

```bash
tensorboard --logdir outputs/tensorboard
```

可查看：

- `Loss/train`、`Loss/val`
- `OA/train`、`OA/val`
- `Macro_F1/train`、`Macro_F1/val`
- `Learning_Rate/downstream`
- `Learning_Rate/resnet`

## 验证与测试

验证集评估：

```bash
python evaluate.py --split val --checkpoint outputs/best_model.pth
```

测试集评估：

```bash
python evaluate.py --split test --checkpoint outputs/best_model.pth
```

也可以直接运行：

```bash
python test.py
```

`test.py` 默认等价于：

```bash
python evaluate.py --split test --checkpoint outputs/best_model.pth
```

## 评估指标

项目使用两个核心指标：

- `OA`：Overall Accuracy，总体分类准确率。
- `macro_f1`：三个类别 F1 分数的宏平均，更适合类别不均衡的分类任务。

训练时以验证集 `macro_f1` 作为保存最优模型的标准。
