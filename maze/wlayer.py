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


class WinfreeOscillatoryLayer(nn.Module):
    """
    Dynamics:
        sensitivity = cos(theta)
        influence   = sin(theta)
        field       = coupling(influence)
        dtheta      = omega + sensitivity * field
    """

    def __init__(
        self,
        ch: int,
        coupling: str = "attn",
        norm: str = "gn",
        rope: bool = True,
        heads: int = 8,
    ):
        super().__init__()

        self.ch = int(ch)
        self.coupling_type = str(coupling)
        self.norm_type = str(norm)
        self.heads = int(heads)

        if self.coupling_type != "attn":
            raise ValueError(
                f"Maze Winfree only supports coupling='attn', "
                f"but got coupling={self.coupling_type!r}."
            )

        if self.norm_type == "gn":
            norm_layer = nn.GroupNorm(pick_gn_groups(self.ch), self.ch)
        elif self.norm_type == "bn":
            norm_layer = nn.BatchNorm2d(self.ch)
        elif self.norm_type in {"none", "identity"}:
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm={self.norm_type!r}.")

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

    def winfree_step(self, theta: torch.Tensor, omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        theta = self.wrap_pm_pi(theta)

        thetas = [] if return_thetas else None
        es = [torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)] if return_es else None

        for _ in range(int(T)):
            dtheta, energy_int = self.winfree_step(theta=theta, omega=omega)

            theta = self.wrap_pm_pi(theta + gamma * dtheta)

            if return_thetas:
                thetas.append(theta)

            if return_es:
                es.append((-energy_int).reshape(theta.shape[0], -1).sum(dim=-1))

        return theta, thetas, es