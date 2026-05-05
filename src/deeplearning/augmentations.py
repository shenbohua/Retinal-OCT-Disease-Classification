from __future__ import annotations

"""Augmentation and preprocessing transforms for OCT deep-learning experiments."""

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, augment: bool = False):
    """Build train/eval transform pipeline with medically safe augmentations."""
    if augment:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.90, 1.00),
                    ratio=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transforms(image_size: int = 224):
    """Compatibility helper for previous module API."""
    return build_transforms(image_size=image_size, augment=True)


def get_eval_transforms(image_size: int = 224):
    """Compatibility helper for previous module API."""
    return build_transforms(image_size=image_size, augment=False)
