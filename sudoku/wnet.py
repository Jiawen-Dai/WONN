from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common.modules import ThetaEmbedding
from sudoku.wlayer import WinfreeOscillatoryLayer

"""
Sudoku WONN. Sudoku uses the trig SI form: sensitivity = cos(theta), influence = sin(theta).
In the reported experiments, L=1 and group_size=1. Therefore, layer transition and group expansion are inactive; the corresponding code is only kept as a compatibility design.
"""


NUM_DIGIT_TOKENS = 10
NUM_OUTPUT_DIGITS = 9


class SudokuInputEmbedding(nn.Module):
    def __init__(self, ch: int, group_size: int = 1):
        super().__init__()

        self.ch = int(ch)
        self.group_size = int(group_size)

        self.omega_template = nn.Parameter(
            torch.randn(
                NUM_DIGIT_TOKENS,
                self.ch,
                self.group_size,
                self.group_size,
            )
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        omega = self.omega_template[tokens.long()]
        omega = omega.permute(0, 3, 1, 4, 2, 5).contiguous()

        b, c, h, g1, w, g2 = omega.shape
        return omega.view(b, c, h * g1, w * g2)


class ThetaUpdate(nn.Module):
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

        u = self.from_cell_grid(self.conv(self.to_cell_grid(torch.cos(theta))))
        v = self.from_cell_grid(self.conv(self.to_cell_grid(torch.sin(theta))))

        radius = torch.sqrt(u * u + v * v + 1e-6)
        u = u / radius
        v = v / radius

        return self.wrap_pm_pi(torch.atan2(v, u))


class OmegaUpdate(nn.Module):
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


class SudokuWinfreeNet(nn.Module):
    def __init__(
        self,
        ch: int = 256,
        L: int = 2,
        T: int | Sequence[int] = 8,
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
        self.group_size = int(group_size)
        self.latent_hw = 9 * self.group_size
        self.norm = str(norm)
        self.heads = int(heads)
        self.coupling = str(coupling)
        self.output_ksize = int(output_ksize)

        if self.coupling != "attn":
            raise ValueError(
                f"SudokuWinfreeNet currently supports coupling='attn', "
                f"but got coupling={self.coupling!r}."
            )

        self.T = self.expand_T(T, self.L)
        self.gamma = nn.Parameter(torch.tensor([float(gamma)]), requires_grad=False)

        self.f_init = SudokuInputEmbedding(ch=self.ch, group_size=self.group_size)
        self.conv0 = self.f_init

        self.layers = self.make_layers()
        self.output = self.make_output(
            ch=self.ch,
            group_size=self.group_size,
            output_ksize=self.output_ksize,
        )
        self.out = nn.Conv2d(
            in_channels=self.ch,
            out_channels=NUM_OUTPUT_DIGITS,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    @property
    def omega_template(self) -> torch.nn.Parameter:
        return self.f_init.omega_template

    @staticmethod
    def expand_T(T: int | Sequence[int], L: int) -> List[int]:
        if isinstance(T, (list, tuple)):
            if len(T) == L:
                return [int(t) for t in T]
            if len(T) == 1:
                return [int(T[0])] * L
            raise ValueError(f"T must be an int, a one-element list, or length L={L}; got len={len(T)}.")

        return [int(T)] * L

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    @staticmethod
    def upsample_mask(mask: torch.Tensor, group_size: int) -> torch.Tensor:
        return mask.repeat_interleave(group_size, dim=2).repeat_interleave(group_size, dim=3)

    @staticmethod
    def make_output(ch: int, group_size: int, output_ksize: int) -> nn.Sequential:
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

    def make_layers(self) -> nn.ModuleList:
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
        inputs: torch.Tensor,
        is_input: torch.Tensor,
        return_thetas: bool = False,
        return_es: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[List[List[torch.Tensor]]],
        Optional[List[List[torch.Tensor]]],
    ]:
        if inputs.dim() != 3:
            raise ValueError(f"inputs must have shape [B, 9, 9], got {tuple(inputs.shape)}")
        if is_input.dim() != 3:
            raise ValueError(f"is_input must have shape [B, 9, 9], got {tuple(is_input.shape)}")

        inputs = inputs.long()
        mask = is_input.unsqueeze(1).float()
        mask_up = self.upsample_mask(mask, self.group_size)

        omega_puzzle = self.f_init(inputs)
        omega_givens = mask_up * omega_puzzle
        omega = omega_puzzle

        theta = self.wrap_pm_pi(0.1 * torch.randn_like(omega))

        thetas: Optional[List[List[torch.Tensor]]] = [] if return_thetas else None
        es: Optional[List[List[torch.Tensor]]] = [] if return_es else None

        for layer_idx, (theta_update, omega_update, winfree_layer) in enumerate(self.layers):
            if layer_idx > 0:
                omega = omega_update(theta, omega) + omega_givens
                theta = theta_update(theta)

            theta, layer_thetas, layer_es = winfree_layer(
                theta=theta,
                omega=omega,
                T=self.T[layer_idx],
                gamma=self.gamma,
                return_thetas=return_thetas,
                return_es=return_es,
            )

            if return_thetas and thetas is not None:
                thetas.append(layer_thetas)

            if return_es and es is not None:
                es.append(layer_es)

        features = self.output(theta)
        return features, thetas, es

    def forward(
        self,
        inputs: torch.Tensor,
        is_input: torch.Tensor,
        return_thetas: bool = False,
        return_es: bool = False,
    ):
        features, thetas, es = self.feature(
            inputs=inputs,
            is_input=is_input,
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