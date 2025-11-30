# models/deeplab_model.py
import torch.nn as nn
import torchvision


def get_deeplabv3_resnet50(num_classes=4, pretrained=True):

    if pretrained:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights="DEFAULT"  # 现在可以安全使用 DEFAULT
        )
    else:
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)

    # 修改分类头
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model


if __name__ == "__main__":
    model = get_deeplabv3_resnet50(num_classes=4)
    print(model)
