from __future__ import annotations

"""Model builders for OCT deep-learning experiments."""

from torch import nn
from torchvision import models


def _freeze_parameters(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """Build one supported backbone and replace classifier head."""
    name = model_name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            _freeze_parameters(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        if freeze_backbone:
            _freeze_parameters(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            _freeze_parameters(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        if freeze_backbone:
            _freeze_parameters(model.features)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    if name in {"mobilenet_v2", "mobilenet", "mobilenetv2"}:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            _freeze_parameters(model.features)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Choose from: resnet18, resnet34, resnet50, vgg16, mobilenet_v2."
    )
