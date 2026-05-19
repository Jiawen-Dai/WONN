from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm

from common.utils import str2bool
from maze.data import (
    MazeDataset,
    IGNORE_INDEX,
    MAZE_OUT_WALL,
    MAZE_OUT_FREE,
    MAZE_OUT_START,
    MAZE_OUT_GOAL,
    MAZE_OUT_PATH,
)
from maze.wnet import MazeWinfreeNet


CLASS_NAMES = {
    MAZE_OUT_WALL: "wall",
    MAZE_OUT_FREE: "free",
    MAZE_OUT_START: "start",
    MAZE_OUT_GOAL: "goal",
    MAZE_OUT_PATH: "path",
}


@dataclass
class MetricCounts:
    board_correct: int = 0
    board_total: int = 0

    wall_correct: int = 0
    wall_total: int = 0

    free_correct: int = 0
    free_total: int = 0

    start_correct: int = 0
    start_total: int = 0

    goal_correct: int = 0
    goal_total: int = 0

    path_correct: int = 0
    path_total: int = 0

    path_tp: int = 0
    path_fp: int = 0
    path_fn: int = 0


def compute_maze_metric_counts(logits: torch.Tensor, target: torch.Tensor) -> MetricCounts:
    pred = logits.argmax(dim=-1)
    valid = target != IGNORE_INDEX

    board_ok = ((pred == target) | ~valid).reshape(target.shape[0], -1).all(dim=1)

    counts = MetricCounts(
        board_correct=int(board_ok.sum().item()),
        board_total=int(target.shape[0]),
    )

    for cls in [MAZE_OUT_WALL, MAZE_OUT_FREE, MAZE_OUT_START, MAZE_OUT_GOAL, MAZE_OUT_PATH]:
        mask = (target == cls) & valid
        total = int(mask.sum().item())
        correct = int(((pred == cls) & mask).sum().item())

        setattr(counts, f"{CLASS_NAMES[cls]}_total", total)
        setattr(counts, f"{CLASS_NAMES[cls]}_correct", correct)

    pred_path = (pred == MAZE_OUT_PATH) & valid
    true_path = (target == MAZE_OUT_PATH) & valid

    counts.path_tp = int((pred_path & true_path).sum().item())
    counts.path_fp = int((pred_path & ~true_path).sum().item())
    counts.path_fn = int((~pred_path & true_path).sum().item())

    return counts


def add_counts_inplace(dst: MetricCounts, src: MetricCounts) -> None:
    for field in dst.__dataclass_fields__.keys():
        setattr(dst, field, int(getattr(dst, field) + getattr(src, field)))


def counts_to_metrics(counts: MetricCounts) -> Dict[str, float]:
    def safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    path_precision = safe_div(counts.path_tp, counts.path_tp + counts.path_fp)
    path_recall = safe_div(counts.path_tp, counts.path_tp + counts.path_fn)

    if path_precision + path_recall > 0:
        path_f1 = 2.0 * path_precision * path_recall / (path_precision + path_recall)
    else:
        path_f1 = 0.0

    return {
        "board_acc": safe_div(counts.board_correct, counts.board_total),
        "wall_acc": safe_div(counts.wall_correct, counts.wall_total),
        "free_acc": safe_div(counts.free_correct, counts.free_total),
        "start_acc": safe_div(counts.start_correct, counts.start_total),
        "goal_acc": safe_div(counts.goal_correct, counts.goal_total),
        "path_acc": safe_div(counts.path_correct, counts.path_total),
        "path_precision": path_precision,
        "path_recall": path_recall,
        "path_f1": path_f1,
    }


def strip_ema_prefix_if_needed(state):
    if isinstance(state, dict) and any(k.startswith("ema_model.") for k in state.keys()):
        return {
            k[len("ema_model."):]: v
            for k, v in state.items()
            if k.startswith("ema_model.")
        }
    return state


def load_state(model: torch.nn.Module, model_path: str) -> None:
    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = strip_ema_prefix_if_needed(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            state = strip_ema_prefix_if_needed(ckpt["state_dict"])
        elif any(k.startswith("ema_model.") for k in ckpt.keys()):
            state = strip_ema_prefix_if_needed(ckpt)
        elif "ema_model" in ckpt and isinstance(ckpt["ema_model"], dict):
            state = strip_ema_prefix_if_needed(ckpt["ema_model"])
        else:
            state = strip_ema_prefix_if_needed(ckpt)
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)


def build_loader(args):
    def worker_init_fn(worker_id: int):
        if args.limit_cores_used:
            os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    dataset = MazeDataset(args.data_root, split=args.split)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    return loader


def unwrap_model_output(out) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    if not isinstance(out, (list, tuple)):
        raise RuntimeError("Energy evaluation requires model(..., return_es=True).")

    logits = out[0]

    if len(out) < 2:
        raise RuntimeError("Model output does not contain energy sequence.")

    es = out[1]

    if es is None:
        raise RuntimeError("Energy sequence is None.")

    return logits, es


def select_layer_energy(es_nested: List[List[torch.Tensor]], energy_layer: int) -> torch.Tensor:
    if len(es_nested) == 0:
        raise RuntimeError("Energy sequence is empty.")

    n_layers = len(es_nested)
    layer_idx = energy_layer if energy_layer >= 0 else n_layers + energy_layer

    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"energy_layer={energy_layer} is out of range for {n_layers} layers.")

    layer_es = es_nested[layer_idx]

    if layer_es is None or len(layer_es) == 0:
        raise RuntimeError(f"Selected layer {layer_idx} has no energy sequence.")

    return torch.stack(layer_es, dim=0).float()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    energy_layer: int = -1,
    num_inits: int = 1,
    vote_mode: str = "both",
):
    model.eval()

    K = max(1, int(num_inits))

    single_counts = MetricCounts()
    final_counts = MetricCounts()
    sum_counts = MetricCounts()

    for X, Y in tqdm.tqdm(loader):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        B = X.shape[0]
        Xk = X.repeat_interleave(K, dim=0)

        out = model(Xk, return_es=True)
        logits_all, es_nested = unwrap_model_output(out)

        logits_bk = logits_all.view(B, K, *logits_all.shape[1:])

        energy_steps = select_layer_energy(es_nested, energy_layer=energy_layer)
        energy_bkt = energy_steps.transpose(0, 1).contiguous().view(B, K, energy_steps.shape[0])

        batch_idx = torch.arange(B, device=device)

        logits_single = logits_bk[:, 0]
        add_counts_inplace(single_counts, compute_maze_metric_counts(logits_single, Y))

        if vote_mode in {"final", "both"}:
            score_final = energy_bkt[:, :, -1]
            best_final_idx = score_final.argmin(dim=1)

            logits_final_vote = logits_bk[batch_idx, best_final_idx]
            add_counts_inplace(final_counts, compute_maze_metric_counts(logits_final_vote, Y))

        if vote_mode in {"sum", "both"}:
            score_sum = energy_bkt[:, :, 1:].sum(dim=-1)
            best_sum_idx = score_sum.argmin(dim=1)

            logits_sum_vote = logits_bk[batch_idx, best_sum_idx]
            add_counts_inplace(sum_counts, compute_maze_metric_counts(logits_sum_vote, Y))

    results = {"single": counts_to_metrics(single_counts)}

    if vote_mode in {"final", "both"}:
        results["vote_final"] = counts_to_metrics(final_counts)

    if vote_mode in {"sum", "both"}:
        results["vote_sum"] = counts_to_metrics(sum_counts)

    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Maze WONN on Maze-Hard with energy voting."
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data/maze-30x30-hard-1k")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--norm", type=str, default="gn")
    parser.add_argument("--coupling", type=str, default="attn", choices=["attn"])
    parser.add_argument("--output_ksize", type=int, default=3)

    parser.add_argument("--num_inits", type=int, default=1)
    parser.add_argument("--vote_mode", type=str, default="both", choices=["final", "sum", "both"])
    parser.add_argument("--energy_layer", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0)
    parser.add_argument("--cpu_core_end", type=int, default=16)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.num_inits < 1:
        raise ValueError("--num_inits must be >= 1")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    device = torch.device("cuda")

    loader = build_loader(args)

    model = MazeWinfreeNet(
        ch=args.ch,
        L=args.L,
        T=args.T,
        coupling=args.coupling,
        gamma=args.gamma,
        group_size=args.group_size,
        norm=args.norm,
        heads=args.heads,
        output_ksize=args.output_ksize,
    ).to(device)

    load_state(model, args.model_path)

    results = evaluate(
        model=model,
        loader=loader,
        device=device,
        energy_layer=args.energy_layer,
        num_inits=args.num_inits,
        vote_mode=args.vote_mode,
    )

    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Model path: {args.model_path}")
    print(f"Num inits per board (K): {args.num_inits}")

    def print_metrics(tag: str, metrics: Dict[str, float]) -> None:
        print(f"\n[{tag}]")
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")

    print_metrics("Single init / k=0", results["single"])

    if "vote_final" in results:
        print_metrics("Energy vote by final step E_T", results["vote_final"])

    if "vote_sum" in results:
        print_metrics("Energy vote by path sum Σ_t E_t", results["vote_sum"])


if __name__ == "__main__":
    main()