from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from common.modules import ThetaEmbedding
from maze.wlayer import WinfreeOscillatoryLayer

"""
Maze WONN. Maze-Hard uses the trig SI form: sensitivity = cos(theta), influence = sin(theta).
In the reported experiments, L=1 and group_size=1. Therefore, layer transition and group expansion are inactive; the corresponding code is only kept as a compatibility design.
"""

MAZE_IN_WALL = 0
MAZE_IN_FREE = 1
MAZE_IN_START = 2
MAZE_IN_GOAL = 3
NUM_INPUT_TOKENS = 4


MAZE_OUT_WALL = 0
MAZE_OUT_FREE = 1
MAZE_OUT_START = 2
MAZE_OUT_GOAL = 3
MAZE_OUT_PATH = 4
NUM_OUTPUT_TOKENS = 5


class MazeInputEmbedding(nn.Module):
    """Map maze input tokens to the initial omega field."""

    def __init__(self, num_tokens: int, ch: int, group_size: int = 1):
        super().__init__()

        self.num_tokens = int(num_tokens)
        self.ch = int(ch)
        self.group_size = int(group_size)

        self.omega_template = nn.Parameter(
            torch.randn(
                self.num_tokens,
                self.ch,
                self.group_size,
                self.group_size,
            )
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        omega = self.omega_template[inp.long()]
        omega = omega.permute(0, 3, 1, 4, 2, 5).contiguous()

        b, c, h, g1, w, g2 = omega.shape

        return omega.view(b, c, h * g1, w * g2)


class ThetaUpdate(nn.Module):
    """Phase transition between Winfree layers."""

    def __init__(
        self,
        ch: int,
        group_size: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()

        self.ch = int(ch)
        self.group_size = int(group_size)
        self.cell_ch = self.ch * self.group_size * self.group_size

        self.conv = nn.Conv2d(
            in_channels=self.cell_ch,
            out_channels=self.cell_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def to_cell_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hg, wg = x.shape

        g = self.group_size
        h = hg // g
        w = wg // g

        x = x.view(b, c, h, g, w, g)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()

        return x.view(b, c * g * g, h, w)

    def from_cell_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, cg2, h, w = x.shape

        g = self.group_size
        c = cg2 // (g * g)

        x = x.view(b, c, g, g, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()

        return x.view(b, c, h * g, w * g)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        theta = self.wrap_pm_pi(theta)

        u = self.conv(self.to_cell_grid(torch.cos(theta)))
        v = self.conv(self.to_cell_grid(torch.sin(theta)))

        u = self.from_cell_grid(u)
        v = self.from_cell_grid(v)

        return self.wrap_pm_pi(torch.atan2(v, u))


class OmegaUpdate(nn.Module):
    """Frequency transition between Winfree layers."""

    def __init__(
        self,
        ch: int,
        group_size: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()

        self.ch = int(ch)
        self.group_size = int(group_size)
        self.cell_ch = self.ch * self.group_size * self.group_size

        self.theta_embedding = ThetaEmbedding(self.ch)

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.cell_ch,
                out_channels=self.cell_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(self.cell_ch),
            nn.ReLU(inplace=True),
        )

    def to_cell_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hg, wg = x.shape

        g = self.group_size
        h = hg // g
        w = wg // g

        x = x.view(b, c, h, g, w, g)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()

        return x.view(b, c * g * g, h, w)

    def from_cell_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, cg2, h, w = x.shape

        g = self.group_size
        c = cg2 // (g * g)

        x = x.view(b, c, g, g, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()

        return x.view(b, c, h * g, w * g)

    def forward(self, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        theta_feat = self.theta_embedding(theta)

        theta_feat = self.to_cell_grid(theta_feat)
        omega_feat = self.to_cell_grid(omega)

        feat = torch.cat([theta_feat, omega_feat], dim=1)
        delta_omega = self.fusion(feat)

        return self.from_cell_grid(delta_omega)


class MazeWinfreeNet(nn.Module):

    def __init__(
        self,
        ch: int = 256,
        L: int = 1,
        T: int = 24,
        coupling: str = "attn",
        gamma: float = 0.1,
        group_size: int = 1,
        norm: str = "gn",
        heads: int = 8,
        output_ksize: int = 3,
    ):
        super().__init__()

        self.ch = int(ch)
        self.L = int(L)
        self.T = int(T)
        self.coupling = str(coupling)
        self.group_size = int(group_size)
        self.norm = str(norm)
        self.heads = int(heads)
        self.output_ksize = int(output_ksize)

        if self.coupling != "attn":
            raise ValueError(
                f"Maze Winfree only supports coupling='attn', "
                f"but got coupling={self.coupling!r}."
            )

        self.gamma = nn.Parameter(torch.tensor([float(gamma)]), requires_grad=False)

        self.f_init = self._make_f_init(
            num_tokens=NUM_INPUT_TOKENS,
            ch=self.ch,
            group_size=self.group_size,
        )
        self.conv0 = self.f_init

        self.layers = self._make_layers()
        self.decoder = self._make_decoder(
            ch=self.ch,
            group_size=self.group_size,
            output_ksize=self.output_ksize,
        )

        self.out = nn.Conv2d(
            in_channels=self.ch,
            out_channels=NUM_OUTPUT_TOKENS,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    @staticmethod
    def _make_f_init(num_tokens: int, ch: int, group_size: int) -> nn.Module:
        return MazeInputEmbedding(
            num_tokens=num_tokens,
            ch=ch,
            group_size=group_size,
        )

    def _make_decoder(self, ch: int, group_size: int, output_ksize: int) -> nn.Sequential:
        padding = output_ksize // 2

        return nn.Sequential(
            ThetaEmbedding(ch),
            nn.Conv2d(
                in_channels=ch,
                out_channels=2 * ch,
                kernel_size=group_size,
                stride=group_size,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(2 * ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=2 * ch,
                out_channels=ch,
                kernel_size=output_ksize,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(ch),
        )

    def _make_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()

        for layer_idx in range(self.L):
            if layer_idx == 0:
                theta_update = nn.Identity()
                omega_update = nn.Identity()
            else:
                theta_update = ThetaUpdate(
                    ch=self.ch,
                    group_size=self.group_size,
                    kernel_size=3,
                    padding=1,
                )
                omega_update = OmegaUpdate(
                    ch=self.ch,
                    group_size=self.group_size,
                    kernel_size=3,
                    padding=1,
                )

            winfree_layer = WinfreeOscillatoryLayer(
                ch=self.ch,
                coupling=self.coupling,
                norm=self.norm,
                rope=True,
                heads=self.heads,
            )

            layers.append(nn.ModuleList([theta_update, omega_update, winfree_layer]))

        return layers

    def feature(
        self,
        inp: torch.Tensor,
        return_thetas: bool = False,
        return_es: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[List[List[torch.Tensor]]],
        Optional[List[List[torch.Tensor]]],
    ]:
        omega_base = self.f_init(inp)
        omega = omega_base

        theta = self.wrap_pm_pi(0.1 * torch.randn_like(omega))

        thetas = [] if return_thetas else None
        es = [] if return_es else None

        for layer_idx, (theta_update, omega_update, winfree_layer) in enumerate(self.layers):
            if layer_idx > 0:
                omega = omega_base + omega_update(theta, omega)
                theta = theta_update(theta)

            theta, layer_thetas, layer_es = winfree_layer(
                theta=theta,
                omega=omega,
                T=self.T,
                gamma=self.gamma,
                return_thetas=return_thetas,
                return_es=return_es,
            )

            if return_thetas:
                thetas.append(layer_thetas)

            if return_es:
                es.append(layer_es)

        features = self.decoder(theta)

        return features, thetas, es

    def forward(
        self,
        inp: torch.Tensor,
        return_thetas: bool = False,
        return_es: bool = False,
    ):
        features, thetas, es = self.feature(
            inp=inp,
            return_thetas=return_thetas,
            return_es=return_es,
        )

        logits = self.out(features)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        if return_thetas or return_es:
            outputs = [logits]

            if return_thetas:
                outputs.append(thetas)

            if return_es:
                outputs.append(es)

            return outputs

        return logits