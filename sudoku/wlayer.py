from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from common.modules import StandardAttention


def pick_gn_groups(ch: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, ch), 0, -1):
        if ch % groups == 0:
            return groups
    return 1


class PatchwiseSingleIFunc(nn.Module):
    def __init__(
        self,
        ch: int,
        group_size: int = 1,
        hidden_ratio: int = 2,
    ):
        super().__init__()

        self.ch = int(ch)
        self.group_size = int(group_size)
        self.hidden_ratio = int(hidden_ratio)

        hidden = self.ch * (self.group_size ** 2) * self.hidden_ratio

        self.op = nn.Sequential(
            nn.Conv2d(
                self.ch,
                hidden,
                kernel_size=self.group_size,
                stride=self.group_size,
                padding=0,
                groups=self.ch,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                self.ch,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.ch,
                bias=True,
            ),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.op(torch.sin(theta))


class WinfreeOscillatoryLayer(nn.Module):
    def __init__(
        self,
        ch: int,
        coupling: str = "attn",
        norm: str = "gn",
        rope: bool = True,
        heads: int = 8,
        group_size: int = 1,
        hidden_ratio: int = 2,
    ):
        super().__init__()

        self.ch = int(ch)
        self.coupling_type = str(coupling)
        self.norm_type = str(norm)
        self.heads = int(heads)
        self.group_size = int(group_size)
        self.hidden_ratio = int(hidden_ratio)

        self.i_func = PatchwiseSingleIFunc(
            ch=self.ch,
            group_size=self.group_size,
            hidden_ratio=self.hidden_ratio,
        )

        if self.coupling_type != "attn":
            raise ValueError(
                f"Sudoku Winfree layer currently supports coupling='attn', "
                f"but got coupling={self.coupling_type!r}."
            )

        if self.norm_type == "gn":
            norm_layer = nn.GroupNorm(pick_gn_groups(self.ch), self.ch)
        elif self.norm_type == "bn":
            norm_layer = nn.BatchNorm2d(self.ch)
        elif self.norm_type in {"none", "identity"}:
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm={self.norm_type!r}. Use 'gn', 'bn', or 'none'.")

        if self.ch % self.heads != 0:
            raise ValueError(f"ch={self.ch} must be divisible by heads={self.heads}.")

        self.coupling = nn.Sequential(
            StandardAttention(
                ch=self.ch,
                heads=self.heads,
                rope=rope,
            ),
            norm_layer,
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def winfree_step(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.wrap_pm_pi(theta)

        sensitivity = torch.cos(theta)
        influence = torch.sin(theta)

        field = self.coupling(influence)
        dtheta = omega + sensitivity * field

        energy_int = influence * field
        return dtheta, energy_int

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        T: int,
        gamma: torch.Tensor,
        return_thetas: bool = False,
        return_es: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        thetas = [] if return_thetas else None
        es = [torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)] if return_es else None

        theta = self.wrap_pm_pi(theta)

        for _ in range(int(T)):
            dtheta, energy_int = self.winfree_step(theta=theta, omega=omega)
            theta = self.wrap_pm_pi(theta + gamma * dtheta)

            if return_thetas:
                thetas.append(theta)

            if return_es:
                es.append((-energy_int).reshape(theta.shape[0], -1).sum(dim=-1))

        return theta, thetas, es
