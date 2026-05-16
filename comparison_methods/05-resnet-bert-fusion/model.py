import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenHFResNetEncoder(nn.Module):
    """本地 HuggingFace ResNet-50 视觉编码器，支持冻结或解冻最后若干 stage。"""
    def __init__(self, model_dir: str, freeze: bool = True, trainable_stages: int = 0) -> None:
        super().__init__()
        # local_files_only=True 保证只从本地目录读取预训练权重，不联网下载。
        self.backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.out_dim = int(self.backbone.config.hidden_sizes[-1])
        self.trainable_stage_modules: list[nn.Module] = []

        if freeze:
            self.freeze_all()
            if trainable_stages > 0:
                self.unfreeze_last_stages(trainable_stages)

    def freeze_all(self) -> None:
        """冻结整个 ResNet backbone。"""
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_stages(self, trainable_stages: int) -> None:
        """只解冻 ResNet encoder 的最后 trainable_stages 个 stage。"""
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
        """重写 train 状态，确保被冻结的 ResNet 部分始终保持 eval。"""
        super().train(mode)

        if not any(param.requires_grad for param in self.backbone.parameters()):
            self.backbone.eval()
            return self

        self.backbone.eval()
        for stage in self.trainable_stage_modules:
            stage.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输入归一化后的图像张量，输出池化后的视觉特征。"""
        grad_enabled = any(param.requires_grad for param in self.backbone.parameters())
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            outputs = self.backbone(pixel_values=image)

        pooled = outputs.pooler_output
        if pooled.ndim > 2:
            pooled = torch.flatten(pooled, start_dim=1)
        return pooled


class BertTextEncoder(nn.Module):
    """使用 BERT 对 POI 文本进行语义编码，并投影到融合所需维度。"""

    def __init__(
        self,
        model_dir: str,
        output_dim: int,
        dropout: float,
        frozen_layers: int = 0,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        # BERT 同样从本地 HuggingFace 目录加载，避免训练时联网。
        self.backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.out_dim = int(self.backbone.config.hidden_size)
        self.frozen_layer_modules: list[nn.Module] = []

        self.freeze_bottom_layers(
            frozen_layers=frozen_layers,
            freeze_embeddings=freeze_embeddings,
        )

        # 将 BERT hidden_size 通常为 768 的句向量投影到统一文本特征维度。
        self.proj = nn.Sequential(
            nn.Linear(self.out_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def freeze_bottom_layers(self, frozen_layers: int, freeze_embeddings: bool = True) -> None:
        """冻结 BERT embedding 层和前 frozen_layers 个 encoder layer。"""
        if freeze_embeddings and hasattr(self.backbone, "embeddings"):
            self.frozen_layer_modules.append(self.backbone.embeddings)

        encoder_layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if encoder_layers is None:
            raise AttributeError("Cannot find backbone.encoder.layer in the loaded BERT model.")

        frozen_layers = max(0, min(frozen_layers, len(encoder_layers)))
        self.frozen_layer_modules.extend(list(encoder_layers[:frozen_layers]))

        for module in self.frozen_layer_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """切换训练/评估状态时，让已冻结的 BERT 模块保持 eval。"""
        super().train(mode)
        for module in self.frozen_layer_modules:
            module.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """输入 tokenizer 结果，输出固定维度的文本语义特征。"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # BERT 有 pooler_output 时优先使用；若模型无 pooler，则退回 CLS token。
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        return self.proj(pooled)


class MultiModalResNetBert(nn.Module):
    """ResNet-50 视觉特征 + BERT 文本语义特征的多模态分类模型。"""

    def __init__(
        self,
        resnet_model_dir: str,
        bert_model_dir: str,
        num_classes: int = 3,
        image_feature_dim: int = 512,
        text_feature_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        freeze_resnet: bool = True,
        resnet_trainable_stages: int = 0,
        bert_frozen_layers: int = 0,
        freeze_bert_embeddings: bool = True,
    ) -> None:
        super().__init__()

        # 图像分支：ResNet-50 输出 pooler 特征，再投影到 image_feature_dim。
        self.image_encoder = FrozenHFResNetEncoder(
            resnet_model_dir,
            freeze=freeze_resnet,
            trainable_stages=resnet_trainable_stages,
        )

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_encoder.out_dim, image_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 文本分支：BERT 输出句子级语义特征，再投影到 text_feature_dim。
        self.text_encoder = BertTextEncoder(
            model_dir=bert_model_dir,
            output_dim=text_feature_dim,
            dropout=dropout,
            frozen_layers=bert_frozen_layers,
            freeze_embeddings=freeze_bert_embeddings,
        )

        # 融合方式为直接拼接，然后使用 MLP 输出 num_classes 个 logits。
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """分别编码图像和文本，拼接两个模态的特征后完成分类。"""
        image_features = self.image_proj(self.image_encoder(image))
        text_features = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused)
