import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenHFResNetEncoder(nn.Module):
    """本地 HuggingFace ResNet-50 视觉编码器，支持全冻结或只解冻最后若干 stage。"""
    def __init__(self, model_dir: str, freeze: bool = True, trainable_stages: int = 0) -> None:
        super().__init__()
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
        """只解冻 ResNet-50 encoder 中最后 trainable_stages 个 stage。"""
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
        super().train(mode)

        if not any(param.requires_grad for param in self.backbone.parameters()):
            self.backbone.eval()
            return self

        # 冻结的前面层保持 eval，只让被解冻的最后 stage 进入 train。
        self.backbone.eval()
        for stage in self.trainable_stage_modules:
            stage.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grad_enabled = any(param.requires_grad for param in self.backbone.parameters())
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            outputs = self.backbone(pixel_values=image)

        pooled = outputs.pooler_output
        if pooled.ndim > 2:
            pooled = torch.flatten(pooled, start_dim=1)
        return pooled


class TextEncoder(nn.Module):
    """将 300 维 Word2Vec 句向量投影到融合所需的文本特征空间。"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, text_feature: torch.Tensor) -> torch.Tensor:
        return self.proj(text_feature)


class MultiModalResNetWord2Vec(nn.Module):
    """ResNet-50 视觉特征 + Word2Vec 文本特征的多模态融合分类模型。"""
    def __init__(
        self,
        word_dim: int,
        resnet_model_dir: str,
        num_classes: int = 3,
        image_feature_dim: int = 512,
        text_feature_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        freeze_resnet: bool = True,
        resnet_trainable_stages: int = 0,
    ) -> None:
        super().__init__()

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

        self.text_encoder = TextEncoder(word_dim, text_feature_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, text_feature: torch.Tensor) -> torch.Tensor:
        image_features = self.image_proj(self.image_encoder(image))
        text_features = self.text_encoder(text_feature)
        fused = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused)
