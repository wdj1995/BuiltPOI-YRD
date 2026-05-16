import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenHFResNetEncoder(nn.Module):
    """本地 HuggingFace ResNet-50 视觉编码器，支持冻结或解冻最后若干 stage。"""

    def __init__(self, model_dir: str, freeze: bool = True, trainable_stages: int = 0) -> None:
        super().__init__()
        # local_files_only=True 保证只从本地目录读取预训练权重。
        self.backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.out_dim = int(self.backbone.config.hidden_sizes[-1])
        self.trainable_stage_modules: list[nn.Module] = []

        if freeze:
            self.freeze_all()
            if trainable_stages > 0:
                self.unfreeze_last_stages(trainable_stages)

    def freeze_all(self) -> None:
        """冻结整个 ResNet-50 backbone。"""
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_stages(self, trainable_stages: int) -> None:
        """只解冻 ResNet encoder 中最后 trainable_stages 个 stage。"""
        stages = getattr(getattr(self.backbone, "encoder", None), "stages", None)
        if stages is None:
            raise AttributeError("Cannot find backbone.encoder.stages in the loaded ResNet model.")

        trainable_stages = min(trainable_stages, len(stages))
        self.trainable_stage_modules = list(stages[-trainable_stages:])

        for stage in self.trainable_stage_modules:
            stage.train()
            for param in stage.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True):
        """重写 train 状态，确保冻结层始终保持 eval。"""
        super().train(mode)

        if not any(param.requires_grad for param in self.backbone.parameters()):
            self.backbone.eval()
            return self

        self.backbone.eval()
        for stage in self.trainable_stage_modules:
            stage.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输出 ResNet 的池化视觉特征。"""
        grad_enabled = any(param.requires_grad for param in self.backbone.parameters())
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            outputs = self.backbone(pixel_values=image)

        pooled = outputs.pooler_output
        if pooled.ndim > 2:
            pooled = torch.flatten(pooled, start_dim=1)
        return pooled


class TextEncoder(nn.Module):
    """使用 BiLSTM 对补齐后的 Word2Vec token 序列进行语义建模。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        # PyTorch LSTM 只有在 num_layers > 1 时才使用内部 dropout。
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        direction_count = 2 if bidirectional else 1
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * direction_count, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, text_feature: torch.Tensor, text_length: torch.Tensor) -> torch.Tensor:
        """输入 [B, max_tokens, word_dim]，输出固定维度文本语义特征。"""
        # pack_padded_sequence 要求 length 在 CPU 上，且不能为 0。
        safe_length = text_length.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            text_feature,
            lengths=safe_length,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            # 双向 LSTM 取最后一层的前向和后向 hidden state 拼接。
            sequence_feature = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            sequence_feature = hidden[-1]
        return self.proj(sequence_feature)


class MultiModalResNetWord2Vec(nn.Module):
    """ResNet-50 视觉特征 + Word2Vec-BiLSTM 文本特征的多模态分类模型。"""

    def __init__(
        self,
        word_dim: int,
        resnet_model_dir: str,
        num_classes: int = 3,
        image_feature_dim: int = 512,
        text_feature_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = True,
        freeze_resnet: bool = True,
        resnet_trainable_stages: int = 0,
    ) -> None:
        super().__init__()

        self.image_encoder = FrozenHFResNetEncoder(
            resnet_model_dir,
            freeze=freeze_resnet,
            trainable_stages=resnet_trainable_stages,
        )

        # 将 ResNet 输出维度投影到统一的视觉特征维度。
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_encoder.out_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 文本分支接收预生成的 Word2Vec token 矩阵，而不是 mean pooling 句向量。
        self.text_encoder = TextEncoder(
            input_dim=word_dim,
            hidden_dim=lstm_hidden_dim,
            output_dim=text_feature_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout=dropout,
        )

        # 直接拼接视觉和文本特征，再通过 MLP 输出三分类 logits。
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, text_feature: torch.Tensor, text_length: torch.Tensor) -> torch.Tensor:
        """前向传播：分别提取两种模态特征，拼接后完成分类。"""
        image_features = self.image_proj(self.image_encoder(image))
        text_features = self.text_encoder(text_feature, text_length)
        fused = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused)
