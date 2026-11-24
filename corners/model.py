"""ResNet-50 backbone + deconvolution head -> 4 corner heatmaps -> soft-argmax (x, y)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .data import STRIDE


class DinoBackbone(nn.Module):
    """DINOv2 ViT-S/14 patch features as a feature map: 256 crop -> resized to 252 -> 18 x 18 x 384."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained, verbose=False)

    def forward(self, x):
        x = F.interpolate(x, size=(252, 252), mode='bilinear', align_corners=False)
        t = self.vit.forward_features(x)['x_norm_patchtokens']             # B x 324 x 384
        B, N, C = t.shape; g = int(N ** 0.5)
        return t.transpose(1, 2).reshape(B, C, g, g).contiguous()


class CornerNet(nn.Module):
    def __init__(self, pretrained=True, n_kp=4, backbone='resnet50'):
        super().__init__()
        self.kind = backbone
        if backbone == 'resnet50':
            r = torchvision.models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            self.backbone = nn.Sequential(*list(r.children())[:-2])      # stride 32, 2048 channels
            layers, c = [], 2048
            for _ in range(3):                                             # stride 32 -> 4 ("simple baselines" head)
                layers += [nn.ConvTranspose2d(c, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
                c = 256
        elif backbone == 'dinov2_vits14':
            self.backbone = DinoBackbone(pretrained)                       # 18 x 18 x 384
            layers = [nn.Conv2d(384, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            for _ in range(2):                                             # 18 -> 72, resized to 64 in forward
                layers += [nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        else:
            raise ValueError(backbone)
        self.head = nn.Sequential(*layers, nn.Conv2d(256, n_kp, 1))
        self.beta = nn.Parameter(torch.tensor(1.0))                        # learned softmax temperature

    def forward(self, x):
        logits = self.head(self.backbone(x))                               # B x K x 64 x 64 (ResNet) / 72 x 72 (DINOv2)
        if logits.shape[-1] != 64:
            logits = F.interpolate(logits, size=(64, 64), mode='bilinear', align_corners=False)
        B, K, H, W = logits.shape
        prob = F.softmax(logits.flatten(2) * self.beta, -1).view(B, K, H, W)
        ys = torch.arange(H, device=x.device, dtype=x.dtype); xs = torch.arange(W, device=x.device, dtype=x.dtype)
        cx = (prob.sum(2) * xs).sum(-1); cy = (prob.sum(3) * ys).sum(-1)
        kp = torch.stack([cx, cy], -1) * STRIDE + (STRIDE - 1) / 2         # heatmap cell -> crop pixel
        return kp, prob


def gaussian_targets(kp, H=64, W=64, sigma=1.5):
    """Normalised Gaussian heatmaps centred on the true corners (in heatmap cells)."""
    c = (kp - (STRIDE - 1) / 2) / STRIDE
    ys = torch.arange(H, device=kp.device).view(1, 1, H, 1); xs = torch.arange(W, device=kp.device).view(1, 1, 1, W)
    g = torch.exp(-((xs - c[..., 0, None, None]) ** 2 + (ys - c[..., 1, None, None]) ** 2) / (2 * sigma ** 2))
    return g / g.sum((2, 3), keepdim=True).clamp_min(1e-12)


def loss_fn(pred_kp, prob, gt_kp, w_hm=50.0):
    """Coordinate L1 (crop px / STRIDE) + heatmap MSE (scaled by the number of cells)."""
    l1 = (pred_kp - gt_kp).abs().mean() / STRIDE
    g = gaussian_targets(gt_kp, prob.shape[2], prob.shape[3])
    mse = ((prob - g) ** 2).sum((2, 3)).mean()
    return l1 + w_hm * mse, l1.detach(), mse.detach()
