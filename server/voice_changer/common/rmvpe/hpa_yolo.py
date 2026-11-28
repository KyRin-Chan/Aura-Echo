import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    """Compute same-padding dynamically for 2D convolution kernels."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> SiLU/Identity block."""

    def __init__(self, c1: int, c2: int, k=1, s=1, p=None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, c1: int, c2: int, k=3, s=1, p=None, act: bool = True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    """Bottleneck block built with depthwise separable convolutions."""

    def __init__(self, c1: int, c2: int, k: int = 3, shortcut: bool = True):
        super().__init__()
        self.dsconv1 = DSConv(c1, c1, k=3, s=1)
        self.dsconv2 = DSConv(c1, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dsconv2(self.dsconv1(x))
        return x + out if self.shortcut else out


class DS_C3k(nn.Module):
    """YOLOv13 CSP block with DS_Bottleneck layers."""

    def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.m = nn.Sequential(
            *[DS_Bottleneck(hidden, hidden, k=k, shortcut=True) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    """YOLOv13 CSP wrapper around DS_C3k."""

    def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.m = DS_C3k(hidden, hidden, n=n, k=k, e=1.0)
        self.cv2 = Conv(hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.m(self.cv1(x)))


class AdaptiveHyperedgeGeneration(nn.Module):
    """Generates soft hypergraph incidence matrices conditioned on global and local context."""

    def __init__(self, in_channels: int, num_hyperedges: int, num_heads: int):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)
        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False)
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        x_perm = x.permute(0, 2, 1)
        pooled = torch.cat(
            (
                F.adaptive_avg_pool1d(x_perm, 1).squeeze(-1),
                F.adaptive_max_pool1d(x_perm, 1).squeeze(-1),
            ),
            dim=1,
        )
        proto = self.global_proto.unsqueeze(0) + self.context_mapper(pooled).view(
            b, self.num_hyperedges, c
        )
        q = self.query_proj(x).view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = proto.view(b, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        attn = (q @ k) * self.scale
        return F.softmax(attn.mean(dim=1).permute(0, 2, 1), dim=-1)


class HypergraphConvolution(nn.Module):
    """Soft hypergraph vertex -> hyperedge -> vertex message passing."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        edge_feat = self.act(self.W_e(A.bmm(x)))
        return x + self.act(self.W_v(A.transpose(1, 2).bmm(edge_feat)))


class AdaptiveHypergraphComputation(nn.Module):
    """Applies adaptive hypergraph generation and convolution over 2D feature maps."""

    def __init__(self, in_channels: int, out_channels: int, num_hyperedges: int, num_heads: int):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(
            in_channels, num_hyperedges, num_heads
        )
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        incidence = self.adaptive_hyperedge_gen(x_flat)
        return self.hypergraph_conv(x_flat, incidence).permute(0, 2, 1).view(b, -1, h, w)


class C3AH(nn.Module):
    """CSP block combining a hypergraph branch with a skip branch."""

    def __init__(self, c1: int, c2: int, num_hyperedges: int, num_heads: int, e: float = 0.5):
        super().__init__()
        hidden = int(c1 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(hidden, hidden, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.ahc(self.cv2(x)), self.cv1(x)), dim=1))


class HyperACE(nn.Module):
    """Hypergraph-based Adaptive Correlation Enhancement module across multi-scale encoder features."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_hyperedges: int = 16,
        num_heads: int = 8,
        k: int = 2,
        l: int = 1,
        c_h: float = 0.5,
        c_l: float = 0.25,
    ):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        c_mid = c4
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1)
        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l

        self.high_order_branch = nn.ModuleList(
            [
                C3AH(self.c_h, self.c_h, num_hyperedges=num_hyperedges, num_heads=num_heads, e=1.0)
                for _ in range(k)
            ]
        )
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)
        self.low_order_branch = nn.Sequential(
            *[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)]
        )
        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        b2, b3, b4, b5 = x
        target_size = b4.shape[2:]
        fused = self.fuse_conv(
            torch.cat(
                (
                    F.interpolate(b2, size=target_size, mode="bilinear", align_corners=False),
                    F.interpolate(b3, size=target_size, mode="bilinear", align_corners=False),
                    b4,
                    F.interpolate(b5, size=target_size, mode="bilinear", align_corners=False),
                ),
                dim=1,
            )
        )
        x_h, x_l, x_s = fused.split([self.c_h, self.c_l, self.c_s], dim=1)
        high_out = self.high_order_fuse(
            torch.cat([m(x_h) for m in self.high_order_branch], dim=1)
        )
        low_out = self.low_order_branch(x_l)
        return self.final_fuse(torch.cat((high_out, low_out, x_s), dim=1))


class GatedFusion(nn.Module):
    """Learnable residual gate fusing hypergraph context into decoder features."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return f_in + self.gamma * h


class YOLO13Encoder(nn.Module):
    """Multi-stage YOLOv13 depthwise-separable encoder producing [P2, P3, P4, P5]."""

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.stem = DSConv(in_channels, base_channels, k=3, s=1)
        self.p2 = nn.Sequential(
            DSConv(base_channels, base_channels * 2, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 2, base_channels * 2, n=1),
        )
        self.p3 = nn.Sequential(
            DSConv(base_channels * 2, base_channels * 4, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 4, base_channels * 4, n=2),
        )
        self.p4 = nn.Sequential(
            DSConv(base_channels * 4, base_channels * 8, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 8, base_channels * 8, n=2),
        )
        self.p5 = nn.Sequential(
            DSConv(base_channels * 8, base_channels * 16, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 16, base_channels * 16, n=1),
        )
        self.out_channels = [
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p2 = self.p2(self.stem(x))
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class YOLO13FullPADDecoder(nn.Module):
    """Top-down FullPAD decoder injecting HyperACE features at each stage."""

    def __init__(self, encoder_channels: list[int], hyperace_out_c: int, out_channels_final: int):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)

        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)
        self.final_conv = Conv(c_d2, out_channels_final, 1, 1)

    def forward(self, enc_feats: list[torch.Tensor], h_ace: torch.Tensor) -> torch.Tensor:
        p2, p3, p4, p5 = enc_feats
        d5 = self.skip_p5(p5)

        d4 = self.up_d5(
            F.interpolate(
                self.fusion_d5(
                    d5,
                    self.h_to_d5(
                        F.interpolate(h_ace, size=d5.shape[2:], mode="bilinear", align_corners=False)
                    ),
                ),
                size=p4.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        ) + self.skip_p4(p4)

        d3 = self.up_d4(
            F.interpolate(
                self.fusion_d4(
                    d4,
                    self.h_to_d4(
                        F.interpolate(h_ace, size=d4.shape[2:], mode="bilinear", align_corners=False)
                    ),
                ),
                size=p3.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        ) + self.skip_p3(p3)

        d2 = self.up_d3(
            F.interpolate(
                self.fusion_d3(
                    d3,
                    self.h_to_d3(
                        F.interpolate(h_ace, size=d3.shape[2:], mode="bilinear", align_corners=False)
                    ),
                ),
                size=p2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        ) + self.skip_p2(p2)

        return self.final_conv(
            self.final_d2(
                self.fusion_d2(
                    d2,
                    self.h_to_d2(
                        F.interpolate(h_ace, size=d2.shape[2:], mode="bilinear", align_corners=False)
                    ),
                )
            )
        )


class HPADeepUnet(nn.Module):
    """YOLOv13-enhanced U-Net backbone for HPA-RMVPE."""

    def __init__(
        self,
        in_channels: int = 1,
        en_out_channels: int = 16,
        base_channels: int = 64,
        hyperace_k: int = 2,
        hyperace_l: int = 1,
        num_hyperedges: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        self.encoder = YOLO13Encoder(in_channels, base_channels)
        enc_ch = self.encoder.out_channels
        self.hyperace = HyperACE(
            in_channels=enc_ch,
            out_channels=enc_ch[-1],
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            k=hyperace_k,
            l=hyperace_l,
        )
        self.decoder = YOLO13FullPADDecoder(
            encoder_channels=enc_ch,
            hyperace_out_c=enc_ch[-1],
            out_channels_final=en_out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        decoded = self.decoder(features, self.hyperace(features))
        return F.interpolate(decoded, size=x.shape[2:], mode="bilinear", align_corners=False)
