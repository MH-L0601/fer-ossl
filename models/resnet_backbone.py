"""ResNet50 骨干 (供 DebGCD 在 ViT 之外可选使用)。

build_resnet50() 返回一个 fc=Identity 的 ResNet50, 其 forward 输出 [B, 2048]
全局池化特征, 与 DebGCD 的 `nn.Sequential(backbone, head)` 完全兼容
(ViT 输出 768 维 cls token, ResNet50 输出 2048 维特征)。

与 fer-ossl 的 model/encoder.py 保持一致: 同样使用 torchvision 的
ResNet50_Weights.DEFAULT (ImageNet 预训练) 作为起点, fc 替换为 Identity,
因此可以用 fer-ossl 保存的 encoder_pretrained_50.pth 直接作为 warmup 权重。
"""
import torch.nn as nn
import torchvision.models as models

FEATURE_DIMS = {
    'vit': 768,
    'resnet50': 2048,
}


def build_resnet50(pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    net = models.resnet50(weights=weights)
    net.fc = nn.Identity()
    return net


def set_trainable_from_grad(backbone, backbone_name, grad_from_block):
    """冻结 backbone 全部参数, 再按 backbone 类型解冻最后若干层。

    Args:
        backbone: 骨干网络。
        backbone_name: 'vit' 或 'resnet50'。
        grad_from_block:
            - vit: 解冻 `blocks.{i}` 中 i >= grad_from_block 的 block
              (DINO 默认 11, 即只微调最后一块)。
            - resnet50: 把 layer1..layer4 映射为 stage 0..3, 解冻
              stage >= min(grad_from_block, 3) 的 stage。
    """
    for m in backbone.parameters():
        m.requires_grad = False

    if backbone_name == 'vit':
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= grad_from_block:
                    m.requires_grad = True
    elif backbone_name in ('resnet50', 'resnet50_clipmap'):
        # layer1..layer4 -> stage 0..3 (resnet50_clipmap wraps resnet50; the mapper stays frozen)
        stage_map = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        threshold = min(grad_from_block, 3)
        for name, m in backbone.named_parameters():
            for stage, idx in stage_map.items():
                if stage in name:
                    if idx >= threshold:
                        m.requires_grad = True
                    break
    else:
        raise ValueError(f'Unknown backbone "{backbone_name}"')
