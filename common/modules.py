import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)


class ThetaEmbedding(nn.Module):
    """Map phase angles theta to learned periodic features."""

    def __init__(self, channels: int, learnable: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.proj = nn.Conv2d(
            in_channels=2 * self.channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.channels,
            bias=True,
        )
        self.proj.requires_grad_(learnable)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x = torch.stack([cos_theta, sin_theta], dim=2)
        x = x.reshape(theta.shape[0], 2 * self.channels, theta.shape[2], theta.shape[3])
        return self.proj(x)


class StandardAttention(nn.Module):
    def __init__(
        self,
        ch: int,
        heads: int = 8,
        rope: bool = True,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        assert ch % heads == 0

        self.ch = int(ch)
        self.heads = int(heads)
        self.head_dim = self.ch // self.heads
        self.rope = bool(rope)
        self.attn_drop_p = float(attn_drop)
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.rope_base = float(rope_base)

        self.W_qkv = nn.Linear(self.ch, 3 * self.ch, bias=qkv_bias)
        self.W_o = nn.Linear(self.ch, self.ch, bias=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(start_dim=-2)

    def _rope_cos_sin_2d(self, H: int, W: int, d: int, device, dtype):
        N = H * W
        cos = torch.ones((N, d), device=device, dtype=dtype)
        sin = torch.zeros((N, d), device=device, dtype=dtype)

        d_y = d // 2
        d_x = d - d_y
        d_y_rot = d_y - (d_y % 2)
        d_x_rot = d_x - (d_x % 2)

        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        if d_y_rot > 0:
            inv_freq_y = 1.0 / (
                self.rope_base ** (torch.arange(0, d_y_rot, 2, device=device, dtype=dtype) / max(d_y_rot, 1))
            )
            freqs_y = torch.outer(yy, inv_freq_y)
            freqs_y = torch.repeat_interleave(freqs_y, repeats=2, dim=-1)
            cos[:, :d_y_rot] = freqs_y.cos()
            sin[:, :d_y_rot] = freqs_y.sin()

        if d_x_rot > 0:
            inv_freq_x = 1.0 / (
                self.rope_base ** (torch.arange(0, d_x_rot, 2, device=device, dtype=dtype) / max(d_x_rot, 1))
            )
            freqs_x = torch.outer(xx, inv_freq_x)
            freqs_x = torch.repeat_interleave(freqs_x, repeats=2, dim=-1)
            start = d_y
            cos[:, start : start + d_x_rot] = freqs_x.cos()
            sin[:, start : start + d_x_rot] = freqs_x.sin()

        return cos, sin

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0)

        scale = 1.0 / math.sqrt(q.shape[-1])
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = torch.softmax(logits, dim=-1)

        if self.attn_drop_p > 0.0:
            attn = F.dropout(attn, p=self.attn_drop_p, training=self.training)

        return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        d = self.head_dim

        x_tokens = x.flatten(2).transpose(1, 2).contiguous()

        qkv = self.W_qkv(x_tokens)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, N, self.heads, d).transpose(1, 2).contiguous()
        k = k.reshape(B, N, self.heads, d).transpose(1, 2).contiguous()
        v = v.reshape(B, N, self.heads, d).transpose(1, 2).contiguous()

        if self.rope:
            cos, sin = self._rope_cos_sin_2d(H, W, d, device=x.device, dtype=q.dtype)
            q, k = self._apply_rope(q, k, cos, sin)

        out = self._sdpa(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C).contiguous()
        out = self.W_o(out)
        out = self.proj_drop(out)

        return out.transpose(1, 2).reshape(B, C, H, W).contiguous()