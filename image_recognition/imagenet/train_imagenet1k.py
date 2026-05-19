import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from common.utils import str2bool, seed_everything
from common.train_utils import (
    log_print,
    make_run_dir,
    ddp_setup,
    ddp_cleanup,
    ddp_barrier,
    seed_for_eval,
    maybe_compile_model,
    save_checkpoint,
    save_ema,
    save_final_models,
    load_finetune,
    load_resume,
)

from image_recognition.wnet import WinfreeOscillatoryNet
from image_recognition.imagenet.data import build_dataloaders, apply_cutmix, IMAGENET_STATS


DATASET = "imagenet1k"


def evaluate(net, dataloader, device, is_main=True, is_ddp=False, log_fh=None):
    net.eval()

    correct = torch.zeros(1, device=device, dtype=torch.long)
    total = torch.zeros(1, device=device, dtype=torch.long)

    iterator = tqdm(dataloader, desc="eval", leave=False) if is_main else dataloader

    with torch.no_grad():
        for inputs, labels in iterator:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = net(inputs)
            pred = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (pred == labels).sum()

    if is_ddp:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    acc = 100.0 * correct.item() / max(total.item(), 1)
    log_print(f"Accuracy of the network on the validation images: {acc:.2f}%", log_fh, is_main)

    return acc


def train_epoch(
    net,
    ema,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    scaler=None,
    grad_clip: float = 0.0,
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 0.8,
    is_main: bool = True,
    is_ddp: bool = False,
    log_fh=None,
):
    net.train()

    running_loss = 0.0
    num_samples = 0
    iterator = tqdm(dataloader, desc=f"epoch {epoch}", leave=False) if is_main else dataloader

    for inputs, labels in iterator:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        inputs, labels_a, labels_b, lam = apply_cutmix(
            inputs,
            labels,
            alpha=cutmix_alpha,
            prob=cutmix_prob,
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp and amp_dtype == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = net(inputs)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        elif use_amp and amp_dtype == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = net(inputs)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

            optimizer.step()

        else:
            logits = net(inputs)
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

            optimizer.step()

        ema.update()

        running_loss += loss.item() * inputs.shape[0]
        num_samples += inputs.shape[0]

    loss_sum = torch.tensor([running_loss], dtype=torch.float64, device=device)
    count_sum = torch.tensor([num_samples], dtype=torch.float64, device=device)

    if is_ddp:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)

    avg_loss = (loss_sum / count_sum.clamp_min(1.0)).item()
    log_print(f"Epoch: {epoch} Training loss: {avg_loss:.3f}", log_fh, is_main)

    return avg_loss


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 10, min_lr: float = 1e-6):
    warmup_epochs = max(int(warmup_epochs), 0)
    cosine_epochs = max(int(epochs) - warmup_epochs, 1)

    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(epochs),
        eta_min=min_lr,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="WONN ImageNet-1K Training")

    parser.add_argument("exp_name", type=str)

    # training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--eval_freq", "--adveval_freq", dest="eval_freq", type=int, default=1)

    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_prob", type=float, default=0.8)

    # data
    parser.add_argument("--data_root", type=str, default="/data/imagenet_common")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--eval_resize", type=int, default=256)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--eval_batchsize", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    # model
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--ch_final", type=int, default=None)

    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--coupling", type=str, default="attn", choices=["attn", "conv"])
    parser.add_argument("--si_func", type=str, default="mlp", choices=["mlp", "trig"])
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[7, 5, 5, 3, 3, 3])

    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--hidden_ratio", type=int, default=2)
    parser.add_argument("--input_patch_size", type=int, default=4)
    parser.add_argument("--output_ksize", type=int, default=3)
    parser.add_argument("--norm", type=str, default="gn", choices=["gn", "bn", "none"])

    # finetune / resume
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ignore_size_mismatch", action="store_true")

    # system
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--deterministic", type=str2bool, default=False)

    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])

    parser.add_argument("--compile", type=str2bool, default=True)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile_backend", type=str, default="inductor")
    parser.add_argument("--compile_dynamic", type=str2bool, default=False)

    return parser


def main():
    args = build_parser().parse_args()

    if args.resume is not None and args.finetune is not None:
        raise ValueError("Use either --resume or --finetune, not both.")

    device, rank, world_size, local_rank, is_ddp = ddp_setup()
    seed_everything(args.seed, deterministic=args.deterministic)

    is_main = (rank == 0)
    jobdir, log_fh = make_run_dir(args.exp_name, root="runs", is_main=is_main)

    log_print(f"[DDP] is_ddp={is_ddp}, world_size={world_size}, device={device}", log_fh, is_main)

    num_classes = IMAGENET_STATS[DATASET]["num_classes"]

    trainloader, testloader, train_sampler, _ = build_dataloaders(
        data=DATASET,
        data_root=args.data_root,
        batch_size=args.batchsize,
        eval_batch_size=args.eval_batchsize,
        workers=args.workers,
        img_size=args.img_size,
        eval_resize=args.eval_resize,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    log_print(
        f"Dataset={DATASET} | train={len(trainloader.dataset)} | "
        f"val={len(testloader.dataset)} | num_classes={num_classes}",
        log_fh,
        is_main,
    )

    model_kwargs = dict(
        data=DATASET,
        out_classes=num_classes,

        ch=args.ch,
        ch_final=args.ch_final,

        L=args.L,
        T=args.T,
        gamma=args.gamma,

        coupling=args.coupling,
        si_func=args.si_func,
        kernel_sizes=args.kernel_sizes,

        group_size=args.group_size,
        hidden_ratio=args.hidden_ratio,
        input_patch_size=args.input_patch_size,
        output_ksize=args.output_ksize,
        norm=args.norm,
    )

    base_net = WinfreeOscillatoryNet(**model_kwargs).to(device)

    if is_main:
        total_params = sum(p.numel() for p in base_net.parameters() if p.requires_grad)
        log_print(f"Total number of parameters: {total_params}", log_fh, is_main)
        log_print(f"Model kwargs: {model_kwargs}", log_fh, is_main)

    optimizer = optim.AdamW(base_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    from ema_pytorch import EMA
    ema = EMA(base_net, beta=args.beta, update_every=1, update_after_step=25)

    load_finetune(
        finetune_path=args.finetune,
        model=base_net,
        optimizer=optimizer,
        ema=ema,
        lr=args.lr,
        ignore_size_mismatch=args.ignore_size_mismatch,
        log_fh=log_fh,
        is_main=is_main,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    use_fp16_scaler = bool(use_amp and args.amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_resume(
            resume_path=args.resume,
            model=base_net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_fp16_scaler else None,
            ema=ema,
            strict_scheduler=True,
            log_fh=log_fh,
            is_main=is_main,
        )

    train_net = maybe_compile_model(base_net, args, device, log_fh=log_fh, is_main=is_main)

    net = train_net
    if is_ddp:
        net = DDP(train_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    writer = SummaryWriter(jobdir) if is_main else None
    writer_ema = SummaryWriter(os.path.join(jobdir, "ema")) if is_main else None

    log_print("Start training...", log_fh, is_main)

    try:
        for epoch in range(start_epoch, args.epochs):
            if is_ddp:
                train_sampler.set_epoch(epoch)

            current_lr = optimizer.param_groups[0]["lr"]

            loss = train_epoch(
                net=net,
                ema=ema,
                dataloader=trainloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                use_amp=use_amp,
                amp_dtype=args.amp_dtype,
                scaler=scaler,
                grad_clip=args.grad_clip,
                cutmix_alpha=args.cutmix_alpha,
                cutmix_prob=args.cutmix_prob,
                is_main=is_main,
                is_ddp=is_ddp,
                log_fh=log_fh,
            )

            if is_main and writer is not None:
                writer.add_scalar("training loss", loss, epoch)
                writer.add_scalar("lr", current_lr, epoch)

            if args.eval_freq > 0 and ((epoch + 1) % args.eval_freq) == 0:
                devices = [local_rank] if device.type == "cuda" else []

                with torch.random.fork_rng(devices=devices):
                    seed_for_eval(args.seed, epoch, device)

                    log_print(f"Evaluating model at epoch {epoch}", log_fh, is_main)
                    acc = evaluate(base_net, testloader, device, is_main=is_main, is_ddp=is_ddp, log_fh=log_fh)

                    log_print(f"Evaluating EMA model at epoch {epoch}", log_fh, is_main)
                    ema_acc = evaluate(ema.ema_model, testloader, device, is_main=is_main, is_ddp=is_ddp, log_fh=log_fh)

                if is_main and writer is not None:
                    writer.add_scalar("model/val accuracy", acc, epoch)

                if is_main and writer_ema is not None:
                    writer_ema.add_scalar("ema/val accuracy", ema_acc, epoch)

            scheduler.step()

            if ((epoch + 1) % args.checkpoint_every) == 0:
                ddp_barrier(is_ddp)

                if is_main:
                    save_checkpoint(
                        base_net,
                        optimizer,
                        epoch,
                        loss,
                        checkpoint_dir=jobdir,
                        scheduler=scheduler,
                        scaler=scaler if use_fp16_scaler else None,
                        args=args,
                    )
                    save_ema(ema, epoch, checkpoint_dir=jobdir)

                ddp_barrier(is_ddp)

        ddp_barrier(is_ddp)

        if is_main:
            save_final_models(base_net, ema, checkpoint_dir=jobdir)
            log_print("Training completed!", log_fh, is_main)

    finally:
        if writer is not None:
            writer.close()
        if writer_ema is not None:
            writer_ema.close()
        if is_main and log_fh is not None:
            log_fh.close()

        ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
