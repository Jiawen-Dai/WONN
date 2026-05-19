from typing import Optional

import torch
import torch.nn as nn

from image_recognition.wlayer import WinfreeOscillatoryLayer
from common.modules import Reshape, ThetaEmbedding


class ThetaUpdate(nn.Module):
    """Phase transition between two Winfree dynamics layers."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # For ImageNet-100, We try bias=True.
        )

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        theta = self.wrap_pm_pi(theta)

        u = self.conv(torch.cos(theta))
        v = self.conv(torch.sin(theta))

        radius = torch.sqrt(u * u + v * v + 1e-6)
        theta_new = torch.atan2(v / radius, u / radius)

        return self.wrap_pm_pi(theta_new)


class OmegaUpdate(nn.Module):
    """Frequency transition between two Winfree dynamics layers."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()

        self.theta_emb = ThetaEmbedding(in_ch)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        theta_feat = self.theta_emb(theta)
        feat = torch.cat([theta_feat, omega], dim=1)
        return self.fusion(feat)


class WinfreeOscillatoryNet(nn.Module):
    """
    WONN for image recognition.

    Main arguments:
        data: cifar10 | cifar100 | imagenet100 | imagenet1k
        ch, ch_final: paper-style channel setting, e.g. Ch=64 -> 256 or Ch=256 -> 256
        coupling: attn | conv
        si_func: mlp | trig
        kernel_sizes: per-layer coupling kernel sizes, when coupling is conv
        group_size: group size N in grouped Winfree dynamics
        input_patch_size: ImageNet-style input patchification size
        output_ksize: output decoder kernel size
    """

    NUM_CLASSES = {
        "cifar10": 10,
        "cifar100": 100,
        "imagenet100": 100,
        "imagenet1k": 1000,
    }

    def __init__(
        self,
        ch: int = 64,
        ch_final: Optional[int] = None,
        out_classes: Optional[int] = None,
        data: str = "cifar100",
        L: int = 6,
        T: int = 3,
        coupling: str = "attn",
        si_func: str = "mlp",
        kernel_sizes=(7, 5, 5, 3, 3, 3),
        gamma: float = 1.0,
        group_size: int = 2,
        hidden_ratio: int = 2,
        input_patch_size: int = 4,
        output_ksize: int = 3,
        norm: str = "gn",
    ):
        super().__init__()

        data = str(data).lower()
        if data not in self.NUM_CLASSES:
            raise ValueError(f"data must be one of {list(self.NUM_CLASSES.keys())}, but got {data!r}.")

        if ch_final is None:
            ch_final = ch

        if out_classes is None:
            out_classes = self.NUM_CLASSES[data]

        self.data = data
        self.L = int(L)
        self.group_size = int(group_size)
        self.hidden_ratio = int(hidden_ratio)
        self.input_patch_size = int(input_patch_size)
        self.output_ksize = int(output_ksize)
        self.norm = str(norm)

        self.channels = self._make_channels(ch=int(ch), ch_final=int(ch_final), L=self.L)
        self.T = self._expand_param(T, self.L)
        self.coupling = self._expand_param(coupling, self.L)
        self.si_func = self._expand_param(si_func, self.L)

        if len(kernel_sizes) != self.L:
            raise ValueError(f"len(kernel_sizes) must be L={self.L}, but got {len(kernel_sizes)}.")
        self.kernel_sizes = [int(k) for k in kernel_sizes]

        self.gamma = nn.Parameter(torch.tensor([float(gamma)]), requires_grad=False)

        self.f_init = self._make_f_init(data=self.data, out_ch=self.channels[0], input_patch_size=self.input_patch_size)
        self.conv0 = self.f_init

        self.layers = self._make_layers()
        self.decoder = self._make_decoder(channels=self.channels[-1], output_ksize=self.output_ksize)

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Reshape(-1, self.channels[-1]))
        self.out = nn.Linear(self.channels[-1], int(out_classes))

    @staticmethod
    def _expand_param(param, length: int):
        if isinstance(param, (list, tuple)):
            param = list(param)
            if len(param) == length:
                return param
            if len(param) == 1:
                return param * length
            raise ValueError(f"Parameter list length must be 1 or {length}, but got {len(param)}.")
        return [param] * length

    @staticmethod
    def _make_channels(ch: int, ch_final: int, L: int):
        """
        Supported paper settings:
            Ch -> Ch
            Ch -> 4Ch

        For Ch -> 4Ch:
            [ch, 2ch, 2ch, 4ch, 4ch, ...]
        """

        if ch_final == ch:
            return [ch] * L

        if ch_final == 4 * ch:
            if L < 4:
                raise ValueError(f"Ch={ch} -> {ch_final} requires L >= 4, but got L={L}.")

            channels = [ch]
            for layer_idx in range(1, L):
                if layer_idx == 1:
                    channels.append(2 * ch)
                elif layer_idx == 3:
                    channels.append(ch_final)
                else:
                    channels.append(channels[-1])
            return channels

        raise ValueError(
            f"Unsupported channel setting: Ch = {ch} -> {ch_final}. "
            f"Use official settings Ch -> Ch or Ch -> 4Ch."
        )

    @staticmethod
    def _make_f_init(data: str, out_ch: int, input_patch_size: int):
        if data in {"cifar10", "cifar100"}:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        return nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=out_ch,
                kernel_size=input_patch_size,
                stride=input_patch_size,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def _make_decoder(self, channels: int, output_ksize: int):
        padding = output_ksize // 2

        return nn.Sequential(
            ThetaEmbedding(channels),
            nn.Conv2d(
                in_channels=channels,
                out_channels=2 * channels,
                kernel_size=output_ksize,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(2 * channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=2 * channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def _make_layers(self):
        layers = nn.ModuleList()

        for layer_idx in range(self.L):
            if layer_idx == 0:
                theta_update = nn.Identity()
                omega_update = nn.Identity()
            else:
                theta_update = ThetaUpdate(in_ch=self.channels[layer_idx - 1], out_ch=self.channels[layer_idx])
                omega_update = OmegaUpdate(in_ch=self.channels[layer_idx - 1], out_ch=self.channels[layer_idx])

            winfree_layer = WinfreeOscillatoryLayer(
                ch=self.channels[layer_idx],
                coupling=self.coupling[layer_idx],
                si_func=self.si_func[layer_idx],
                kernel_size=self.kernel_sizes[layer_idx],
                group_size=self.group_size,
                norm=self.norm,
                hidden_ratio=self.hidden_ratio,
                rope=True,
            )

            layers.append(nn.ModuleList([theta_update, omega_update, winfree_layer]))

        return layers

    def feature(self, inp: torch.Tensor, return_thetas: bool = False, return_es: bool = False):
        omega = self.f_init(inp)
        theta = self.wrap_pm_pi(0.01 * torch.randn_like(omega))

        thetas = [] if return_thetas else None
        es = [] if return_es else None

        for layer_idx, (theta_update, omega_update, winfree_layer) in enumerate(self.layers):
            if layer_idx > 0:
                omega = omega_update(theta, omega)
                theta = theta_update(theta)

            theta, layer_thetas, layer_es = winfree_layer(
                theta,
                omega,
                self.T[layer_idx],
                self.gamma,
                return_thetas=return_thetas,
                return_es=return_es,
            )

            if return_thetas:
                thetas.append(layer_thetas)

            if return_es:
                es.append(layer_es)

        decoded = self.decoder(theta)
        theta_pooled, decoded_pooled = map(self.pool, (theta, decoded))

        return decoded_pooled, theta_pooled, thetas, es

    def forward(self, inp: torch.Tensor, return_thetas: bool = False, return_es: bool = False):
        features, theta, thetas, es = self.feature(inp, return_thetas=return_thetas, return_es=return_es)
        logits = self.out(features)

        if return_thetas or return_es:
            outputs = [logits]
            if return_thetas:
                outputs.append(thetas)
            if return_es:
                outputs.append(es)
            return outputs

        return logits