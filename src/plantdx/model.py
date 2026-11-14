"""Model definition and image transforms shared by training, evaluation and export."""
import torch.nn as nn
import torchvision
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def build_model(num_classes, pretrained=True):
    """MobileNetV3-Large with a new classification head. Small enough for phones."""
    weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def train_transform():
    # Colour jitter and random crops make the model less tied to lab backgrounds.
    return T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
