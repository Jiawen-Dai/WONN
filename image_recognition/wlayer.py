import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules import StandardAttention


class TrigSFunc(nn.Module):
    """S(theta) = cos(theta)."""

    def __init__(self, ch: int):
        super().__init__()
        self.ch = int(ch)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        assert theta.shape[1] == self.ch, f"Expected C={self.ch}, got C={theta.shape[1]}"
        return torch.cos(theta)


class MlpSFunc(nn.Module):
    """Learnable pointwise S(theta) from [sin(theta), cos(theta)]."""

    def __init__(self, ch: int, hidden_ratio: int = 2):
        super().__init__()
        self.ch = int(ch)
        hidden = self.ch * int(hidden_ratio)

        self.net = nn.Sequential(
            nn.Conv2d(2 * self.ch, hidden, kernel_size=1, groups=self.ch, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.ch, kernel_size=1, groups=self.ch, bias=True),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        b, c, h, w = theta.shape
        assert c == self.ch, f"Expected C={self.ch}, got C={c}"

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        emb = torch.stack([sin_theta, cos_theta], dim=2).reshape(b, 2 * c, h, w)

        return self.net(emb)


class TrigIFunc(nn.Module):
    """Group-wise I(theta) from sin(theta)."""

    def __init__(self, ch: int, group_size: int = 2, hidden_ratio: int = 2):
        super().__init__()
        self.ch = int(ch)
        self.group_size = int(group_size)

        hidden = self.ch * (self.group_size ** 2) * int(hidden_ratio)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=self.ch,
                out_channels=hidden,
                kernel_size=self.group_size,
                stride=self.group_size,
                groups=self.ch,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.ch, kernel_size=1, groups=self.ch, bias=True),
        )

    def pad(self, x: torch.Tensor):
        p = self.group_size
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return x, h, w

    def forward(self, theta: torch.Tensor):
        assert theta.shape[1] == self.ch, f"Expected C={self.ch}, got C={theta.shape[1]}"

        emb = torch.sin(theta)
        emb, h, w = self.pad(emb)
        influence = self.net(emb)

        return influence, h, w


class MlpIFunc(nn.Module):
    """Group-wise learnable I(theta) from [sin(theta), cos(theta)]."""

    def __init__(self, ch: int, group_size: int = 2, hidden_ratio: int = 2):
        super().__init__()
        self.ch = int(ch)
        self.group_size = int(group_size)

        hidden = self.ch * (self.group_size ** 2) * 2 * int(hidden_ratio)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * self.ch,
                out_channels=hidden,
                kernel_size=self.group_size,
                stride=self.group_size,
                groups=self.ch,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.ch, kernel_size=1, groups=self.ch, bias=True),
        )

    def pad(self, x: torch.Tensor):
        p = self.group_size
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        return x, h, w

    def forward(self, theta: torch.Tensor):
        b, c, h, w = theta.shape
        assert c == self.ch, f"Expected C={self.ch}, got C={c}"

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        emb = torch.stack([sin_theta, cos_theta], dim=2).reshape(b, 2 * c, h, w)

        emb, h, w = self.pad(emb)
        influence = self.net(emb)

        return influence, h, w


class WinfreeOscillatoryLayer(nn.Module):
    def __init__(
        self,
        ch: int,
        coupling: str = "attn",
        si_func: str = "mlp",
        norm: str = "gn",
        kernel_size: int = 3,
        group_size: int = 2,
        hidden_ratio: int = 2,
        rope: bool = True,
        heads: int = 8,
    ):
        super().__init__()

        self.ch = int(ch)
        self.group_size = int(group_size)

        if si_func == "mlp":
            self.S = MlpSFunc(ch=self.ch, hidden_ratio=hidden_ratio)
            self.I = MlpIFunc(ch=self.ch, group_size=group_size, hidden_ratio=hidden_ratio)
        elif si_func == "trig":
            self.S = TrigSFunc(ch=self.ch)
            self.I = TrigIFunc(ch=self.ch, group_size=group_size, hidden_ratio=hidden_ratio)
        else:
            raise ValueError(f"Unsupported si_func={si_func!r}. Use 'mlp' or 'trig'.")

        if norm == "gn":
            norm_layer = nn.GroupNorm(8 if self.ch % 8 == 0 else 1, self.ch)
        elif norm == "bn":
            norm_layer = nn.BatchNorm2d(self.ch)
        elif norm == "none":
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm={norm!r}. Use 'gn', 'bn', or 'none'.")

        if coupling == "attn":
            assert self.ch % heads == 0, f"ch={self.ch} must be divisible by heads={heads}"
            self.coupling = nn.Sequential(
                StandardAttention(ch=self.ch, heads=heads, rope=rope),
                norm_layer,
                nn.ReLU(inplace=True),
            )
        elif coupling == "conv":
            self.coupling = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.ch,
                    out_channels=self.ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                norm_layer,
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported coupling={coupling!r}. Use 'attn' or 'conv'.")

    @staticmethod
    def wrap_pm_pi(theta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(theta), torch.cos(theta))

    def broadcast(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        p = self.group_size
        x = x.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return x[:, :, :h, :w]

    def winfree_step(self, theta: torch.Tensor, omega: torch.Tensor):
        theta = self.wrap_pm_pi(theta)

        sensitivity = self.S(theta)
        influence, h, w = self.I(theta)

        field_group = self.coupling(influence)
        field = self.broadcast(field_group, h, w)

        dtheta = omega + sensitivity * field
        energy_int = field_group * influence

        return dtheta, energy_int

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        T: int,
        gamma: torch.Tensor,
        return_thetas: bool = False,
        return_es: bool = False,
    ):
        thetas = [] if return_thetas else None
        es = [torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)] if return_es else None

        theta = self.wrap_pm_pi(theta)

        for _ in range(int(T)):
            dtheta, energy_int = self.winfree_step(theta, omega)
            theta = self.wrap_pm_pi(theta + gamma * dtheta)

            if return_thetas:
                thetas.append(theta)

            if return_es:
                es.append((-energy_int).reshape(theta.shape[0], -1).sum(-1))

        return theta, thetas, es