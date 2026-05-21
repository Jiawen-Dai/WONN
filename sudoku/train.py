from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from common.train_utils import (
    ddp_barrier,
    ddp_cleanup,
    ddp_setup,
    load_finetune,
    load_resume,
    log_print,
    make_run_dir,
    maybe_compile_model,
    save_checkpoint,
    save_ema,
    save_final_models,
)
from common.utils import seed_everything, str2bool
from sudoku.data import build_dataloaders, move_batch_to_device
from sudoku.wnet import SudokuWinfreeNet


Metrics = Dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser(description="Train WONN on Sudoku.")

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="runs/sudoku")
    parser.add_argument("--data_root", type=str, default="./data/sudoku")

    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--eval_batchsize", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.995)
    parser.add_argument("--ema_update_every", type=int, default=10)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--given_loss_weight", type=float, default=1.0)
    parser.add_argument("--blank_loss_weight", type=float, default=1.0)

    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--ch", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--coupling", type=str, default="attn")
    parser.add_argument("--norm", type=str, default="gn")
    parser.add_argument("--output_ksize", type=int, default=3)

    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--ignore_size_mismatch", type=str2bool, default=False)

    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile_backend", type=str, default="inductor")
    parser.add_argument("--compile_dynamic", type=str2bool, default=False)

    return parser.parse_args()


def amp_dtype_from_name(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype={name!r}.")


def ddp_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def weighted_sudoku_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_input: torch.Tensor,
    given_loss_weight: float,
    blank_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = logits.shape[0]

    per_cell_loss = F.cross_entropy(
        logits.reshape(-1, 9),
        targets.reshape(-1),
        reduction="none",
    ).view(batch_size, 9, 9)

    given_mask = is_input.float()
    blank_mask = 1.0 - given_mask
    weights = given_loss_weight * given_mask + blank_loss_weight * blank_mask

    loss = (per_cell_loss * weights).sum() / weights.sum().clamp_min(1.0)
    given_loss = (per_cell_loss * given_mask).sum() / given_mask.sum().clamp_min(1.0)
    blank_loss = (per_cell_loss * blank_mask).sum() / blank_mask.sum().clamp_min(1.0)

    return loss, given_loss.detach(), blank_loss.detach()


def forward_with_energy(model, inputs: torch.Tensor, is_input: torch.Tensor):
    outputs = model(inputs, is_input, return_es=True)
    logits = outputs[0]
    energies = outputs[1]
    final_energy = energies[-1][-1].mean()
    return logits, final_energy


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Metrics:
    model.eval()

    totals = torch.zeros(8, device=device, dtype=torch.float64)

    for batch in loader:
        inputs, targets, is_input = move_batch_to_device(batch, device)
        logits = model(inputs, is_input)

        loss_sum = F.cross_entropy(
            logits.reshape(-1, 9),
            targets.reshape(-1),
            reduction="sum",
        )

        pred = logits.argmax(dim=-1)
        correct = pred.eq(targets)
        given_mask = is_input.bool()
        blank_mask = ~given_mask

        totals[0] += loss_sum.double()
        totals[1] += float(targets.numel())
        totals[2] += correct.reshape(correct.shape[0], -1).all(dim=1).sum().double()
        totals[3] += float(targets.shape[0])
        totals[4] += (correct & given_mask).sum().double()
        totals[5] += given_mask.sum().double()
        totals[6] += (correct & blank_mask).sum().double()
        totals[7] += blank_mask.sum().double()

    totals = ddp_reduce_sum(totals)

    loss = totals[0] / totals[1].clamp_min(1.0)
    board_acc = totals[2] / totals[3].clamp_min(1.0)
    givens_acc = totals[4] / totals[5].clamp_min(1.0)
    blanks_acc = totals[6] / totals[7].clamp_min(1.0)

    return {
        "loss": float(loss.item()),
        "board_acc": float(board_acc.item()),
        "givens_acc": float(givens_acc.item()),
        "blanks_acc": float(blanks_acc.item()),
    }


def log_metrics(writer, metrics: Metrics, prefix: str, epoch: int):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)


def main():
    args = parse_args()

    if args.resume is not None and args.finetune is not None:
        raise ValueError("Use either --resume or --finetune, not both.")

    seed_everything(args.seed, deterministic=args.deterministic)
    device, rank, world_size, local_rank, is_ddp = ddp_setup()
    is_main = rank == 0

    jobdir, log_fh = make_run_dir(args.exp_name, root=args.save_dir, is_main=is_main)
    writer = SummaryWriter(jobdir) if is_main else None

    try:
        log_print(
            f"[DDP] is_ddp={is_ddp}, world_size={world_size}, device={device}",
            log_fh,
            is_main,
        )

        trainloader, testloader, train_sampler, _ = build_dataloaders(
            data_root=args.data_root,
            batch_size=args.batchsize,
            eval_batch_size=args.eval_batchsize,
            num_workers=args.num_workers,
            is_ddp=is_ddp,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )

        log_print(
            f"Sudoku dataset | train={len(trainloader.dataset)} | test={len(testloader.dataset)}",
            log_fh,
            is_main,
        )

        base_model = SudokuWinfreeNet(
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

        num_params = sum(p.numel() for p in base_model.parameters())
        log_print(f"Total number of parameters: {num_params}", log_fh, is_main)

        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        ema = EMA(
            base_model,
            beta=args.beta,
            update_every=args.ema_update_every,
            update_after_step=args.ema_update_after_step,
        ).to(device)

        use_amp = bool(args.amp and device.type == "cuda")
        use_fp16_scaler = bool(use_amp and args.amp_dtype == "fp16")
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

        if args.finetune is not None:
            load_finetune(
                args.finetune,
                base_model,
                optimizer=optimizer,
                ema=ema,
                lr=args.lr,
                ignore_size_mismatch=args.ignore_size_mismatch,
                log_fh=log_fh,
                is_main=is_main,
            )

        start_epoch = load_resume(
            args.resume,
            base_model,
            optimizer,
            scaler=scaler,
            ema=ema,
            strict_scheduler=False,
            log_fh=log_fh,
            is_main=is_main,
        )

        train_model = base_model
        if is_ddp:
            train_model = DDP(
                base_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        train_model = maybe_compile_model(train_model, args, device, log_fh=log_fh, is_main=is_main)

        amp_dtype = amp_dtype_from_name(args.amp_dtype)

        for epoch in range(start_epoch, args.epochs):
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_model.train()
            ema.train()

            train_totals = torch.zeros(5, device=device, dtype=torch.float64)
            pbar = tqdm.tqdm(trainloader, disable=not is_main, desc=f"epoch {epoch}")

            for batch in pbar:
                inputs, targets, is_input = move_batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits, energy = forward_with_energy(train_model, inputs, is_input)
                        loss, given_loss, blank_loss = weighted_sudoku_loss(
                            logits=logits,
                            targets=targets,
                            is_input=is_input,
                            given_loss_weight=args.given_loss_weight,
                            blank_loss_weight=args.blank_loss_weight,
                        )

                    if use_fp16_scaler:
                        scaler.scale(loss).backward()
                        if args.clip_grad_norm > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.clip_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if args.clip_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.clip_grad_norm)
                        optimizer.step()
                else:
                    logits, energy = forward_with_energy(train_model, inputs, is_input)
                    loss, given_loss, blank_loss = weighted_sudoku_loss(
                        logits=logits,
                        targets=targets,
                        is_input=is_input,
                        given_loss_weight=args.given_loss_weight,
                        blank_loss_weight=args.blank_loss_weight,
                    )

                    loss.backward()
                    if args.clip_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.clip_grad_norm)
                    optimizer.step()

                ema.update()

                train_totals[0] += float(loss.item())
                train_totals[1] += float(given_loss.item())
                train_totals[2] += float(blank_loss.item())
                train_totals[3] += float(energy.item())
                train_totals[4] += 1.0

                if is_main:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_totals = ddp_reduce_sum(train_totals)
            num_steps = train_totals[4].clamp_min(1.0)

            train_metrics = {
                "loss": float((train_totals[0] / num_steps).item()),
                "givens_loss": float((train_totals[1] / num_steps).item()),
                "blanks_loss": float((train_totals[2] / num_steps).item()),
                "final_energy": float((train_totals[3] / num_steps).item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            if is_main:
                log_metrics(writer, train_metrics, "train", epoch)
                log_print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"loss={train_metrics['loss']:.6f} "
                    f"givens_loss={train_metrics['givens_loss']:.6f} "
                    f"blanks_loss={train_metrics['blanks_loss']:.6f} "
                    f"energy={train_metrics['final_energy']:.6f}",
                    log_fh,
                    is_main,
                )

            if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
                metrics = evaluate(base_model, testloader, device)
                ema_metrics = evaluate(ema.ema_model, testloader, device)

                if is_main:
                    log_metrics(writer, metrics, "test", epoch)
                    log_metrics(writer, ema_metrics, "ema_test", epoch)
                    log_print(
                        f"[Test] board_acc={metrics['board_acc']:.6f} "
                        f"givens_acc={metrics['givens_acc']:.6f} "
                        f"blanks_acc={metrics['blanks_acc']:.6f}",
                        log_fh,
                        is_main,
                    )
                    log_print(
                        f"[EMA Test] board_acc={ema_metrics['board_acc']:.6f} "
                        f"givens_acc={ema_metrics['givens_acc']:.6f} "
                        f"blanks_acc={ema_metrics['blanks_acc']:.6f}",
                        log_fh,
                        is_main,
                    )

            if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
                if is_main:
                    save_checkpoint(
                        base_model,
                        optimizer,
                        epoch,
                        train_metrics["loss"],
                        checkpoint_dir=jobdir,
                        scaler=scaler,
                        args=args,
                    )
                    save_ema(ema, epoch, checkpoint_dir=jobdir)

            ddp_barrier(is_ddp)

        if is_main:
            save_final_models(base_model, ema, jobdir)

    finally:
        if writer is not None:
            writer.close()
        if log_fh is not None:
            log_fh.close()
        ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
