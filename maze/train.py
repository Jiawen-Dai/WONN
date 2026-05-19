from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm

from common.train_utils import ddp_cleanup, ddp_setup, maybe_compile_model, save_checkpoint, save_ema
from common.utils import str2bool
from maze.data import (
    build_dataloaders,
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


def reduce_scalar(value: float, device: torch.device, average: bool = True) -> float:
    x = torch.tensor(value, dtype=torch.float64, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        if average:
            x /= dist.get_world_size()

    return float(x.item())


def all_reduce_counts(counts: MetricCounts, device: torch.device) -> MetricCounts:
    fields = list(counts.__dataclass_fields__.keys())

    arr = torch.tensor(
        [getattr(counts, field) for field in fields],
        dtype=torch.float64,
        device=device,
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(arr, op=dist.ReduceOp.SUM)

    vals = arr.cpu().long().tolist()

    return MetricCounts(**{field: int(value) for field, value in zip(fields, vals)})


def add_counts_inplace(dst: MetricCounts, src: MetricCounts) -> None:
    for field in dst.__dataclass_fields__.keys():
        setattr(dst, field, int(getattr(dst, field) + getattr(src, field)))


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


@torch.no_grad()
def compute_acc_ddp(net: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    net.eval()

    counts = MetricCounts()

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        logits = net(X)
        add_counts_inplace(counts, compute_maze_metric_counts(logits, Y))

    counts = all_reduce_counts(counts, device=device)

    return counts_to_metrics(counts)


def forward_with_energy(net: nn.Module, X: torch.Tensor):
    outputs = net(X, return_es=True)

    logits = outputs[0]
    es = outputs[1]

    final_energy = es[-1][-1].mean()

    return logits, final_energy


def build_class_weights(args, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [args.w_wall, args.w_free, args.w_start, args.w_goal, args.w_path],
        dtype=torch.float32,
        device=device,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Train Maze WONN on Maze-Hard 30x30.")

    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=9000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.995)
    parser.add_argument("--grad_clip", "--clip_grad_norm", dest="grad_clip", type=float, default=0.0)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--ema_update_every", type=int, default=10)
    parser.add_argument("--ema_update_after_step", type=int, default=100)

    parser.add_argument("--data_root", type=str, default="./data/maze-30x30-hard-1k")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num_workers", "--workers", dest="num_workers", type=int, default=4)

    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0)
    parser.add_argument("--cpu_core_end", type=int, default=16)

    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--norm", type=str, default="gn")
    parser.add_argument("--coupling", type=str, default="attn", choices=["attn"])
    parser.add_argument("--output_ksize", type=int, default=3)

    parser.add_argument("--w_wall", type=float, default=1.0)
    parser.add_argument("--w_free", type=float, default=5.0)
    parser.add_argument("--w_start", type=float, default=2.0)
    parser.add_argument("--w_goal", type=float, default=2.0)
    parser.add_argument("--w_path", type=float, default=5.0)

    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])

    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile_backend", type=str, default="inductor")
    parser.add_argument("--compile_dynamic", type=str2bool, default=False)

    parser.add_argument("--save_dir", type=str, default=os.path.join("runs", "maze"))
    parser.add_argument("--speed_test", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device, rank, world_size, local_rank, use_ddp = ddp_setup()
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main:
        print("Exp name:", args.exp_name, flush=True)
        print(f"DDP enabled: {use_ddp} | world_size={world_size}", flush=True)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(enabled=True)

    if args.seed is not None:
        seed = args.seed + rank
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    if args.limit_cores_used:
        def worker_init_fn(worker_id: int):
            os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))
    else:
        worker_init_fn = None

    trainloader, testloader, train_sampler, _ = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batchsize,
        test_batch_size=100,
        num_workers=args.num_workers,
        is_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        prefetch_factor=4,
    )

    if is_main:
        print(f"Data root: {args.data_root}", flush=True)
        print(f"Train size: {len(trainloader.dataset)} | Test size: {len(testloader.dataset)}", flush=True)

    jobdir = os.path.join(args.save_dir, args.exp_name)
    if is_main:
        os.makedirs(jobdir, exist_ok=True)

    base_net = MazeWinfreeNet(
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

    if is_main:
        total_params = sum(p.numel() for p in base_net.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}", flush=True)

    model_for_train = base_net

    if use_ddp:
        model_for_train = DDP(
            base_net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    net = maybe_compile_model(model_for_train, args, device, is_main=is_main)

    optimizer = torch.optim.Adam(base_net.parameters(), lr=args.lr)

    scheduler1 = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=8800,
    )

    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[8800],
    )

    ema = EMA(
        base_net,
        beta=args.beta,
        update_every=args.ema_update_every,
        update_after_step=args.ema_update_after_step,
    )

    class_weights = build_class_weights(args, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )

    use_amp = bool(args.amp and torch.cuda.is_available())
    use_fp16_scaler = bool(use_amp and args.amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    if args.speed_test:
        it_sp = 0
        time_per_iter = []

    try:
        for epoch in range(args.epochs):
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            total_loss = 0.0
            total_energy = 0.0

            net.train()
            ema.train()

            pbar = tqdm.tqdm(trainloader, disable=not is_main)

            for X, Y in pbar:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                if args.speed_test:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                optimizer.zero_grad(set_to_none=True)

                if use_amp and args.amp_dtype == "fp16":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits, energy = forward_with_energy(net, X)
                        loss_map = criterion(logits.permute(0, 3, 1, 2), Y)
                        valid = (Y != IGNORE_INDEX).float()
                        loss = (loss_map * valid).sum() / valid.sum().clamp_min(1.0)

                    scaler.scale(loss).backward()

                    if args.grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(base_net.parameters(), args.grad_clip)

                    scaler.step(optimizer)
                    scaler.update()

                elif use_amp and args.amp_dtype == "bf16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, energy = forward_with_energy(net, X)
                        loss_map = criterion(logits.permute(0, 3, 1, 2), Y)
                        valid = (Y != IGNORE_INDEX).float()
                        loss = (loss_map * valid).sum() / valid.sum().clamp_min(1.0)

                    loss.backward()

                    if args.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(base_net.parameters(), args.grad_clip)

                    optimizer.step()

                else:
                    logits, energy = forward_with_energy(net, X)
                    loss_map = criterion(logits.permute(0, 3, 1, 2), Y)
                    valid = (Y != IGNORE_INDEX).float()
                    loss = (loss_map * valid).sum() / valid.sum().clamp_min(1.0)

                    loss.backward()

                    if args.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(base_net.parameters(), args.grad_clip)

                    optimizer.step()

                ema.update()

                if args.speed_test:
                    end.record()
                    torch.cuda.synchronize()
                    time_elapsed_per_iter = start.elapsed_time(end)
                    time_per_iter.append(time_elapsed_per_iter)

                    if is_main:
                        print(time_elapsed_per_iter, flush=True)

                    it_sp += 1

                    if it_sp == 100:
                        if is_main:
                            np.save(os.path.join(jobdir, "time.npy"), np.array(time_per_iter))
                        raise SystemExit(0)

                total_loss += float(loss.item())
                total_energy += float(energy.item())

            total_loss /= max(len(trainloader), 1)
            total_energy /= max(len(trainloader), 1)

            total_loss = reduce_scalar(total_loss, device=device, average=True)
            total_energy = reduce_scalar(total_energy, device=device, average=True)

            if is_main:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss:.6f}, FinalEnergy: {total_energy:.6f}",
                    flush=True,
                )

            if (epoch + 1) % args.eval_freq == 0:
                metrics = compute_acc_ddp(base_net, testloader, device)
                ema_metrics = compute_acc_ddp(ema.ema_model, testloader, device)

                if is_main:
                    print("[Test]", {k: round(v, 6) for k, v in metrics.items()}, flush=True)
                    print("[EMA Test]", {k: round(v, 6) for k, v in ema_metrics.items()}, flush=True)

            if (epoch + 1) % args.checkpoint_every == 0 and is_main:
                save_checkpoint(base_net, optimizer, epoch, total_loss, checkpoint_dir=jobdir, latest=False)
                save_ema(ema, epoch, checkpoint_dir=jobdir, latest=False)

            scheduler.step()

            if use_ddp:
                dist.barrier()

        if is_main:
            torch.save(base_net.state_dict(), os.path.join(jobdir, "model.pth"))
            torch.save(ema.state_dict(), os.path.join(jobdir, "ema_model.pth"))

    finally:
        ddp_cleanup(use_ddp)


if __name__ == "__main__":
    main()