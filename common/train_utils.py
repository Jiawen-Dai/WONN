import os
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist

from common.utils import load_state_dict_ignore_size_mismatch


def log_print(msg, log_fh=None, is_main=True):
    if not is_main:
        return

    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


def make_run_dir(exp_name: str, root: str = "runs", is_main: bool = True):
    jobdir = os.path.join(root, exp_name)
    log_fh = None

    if is_main:
        os.makedirs(jobdir, exist_ok=True)
        log_fh = open(os.path.join(jobdir, "train.txt"), "a", buffering=1, encoding="utf-8")

    return jobdir, log_fh


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

        device = torch.device("cuda", local_rank)
        is_ddp = True

    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_ddp = False

    return device, rank, world_size, local_rank, is_ddp


def ddp_cleanup(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()


def seed_for_eval(seed: int, epoch: int, device: torch.device):
    seed = int(seed) + 100000 + int(epoch)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def maybe_compile_model(model, args, device, log_fh=None, is_main=True):
    if not args.compile:
        return model

    if device.type != "cuda":
        log_print("torch.compile requested but device is not CUDA; skip compile.", log_fh, is_main)
        return model

    if not hasattr(torch, "compile"):
        log_print("torch.compile is unavailable in this PyTorch version; skip compile.", log_fh, is_main)
        return model

    kwargs = {"backend": args.compile_backend, "mode": args.compile_mode}

    try:
        kwargs["dynamic"] = bool(args.compile_dynamic)
        model = torch.compile(model, **kwargs)
    except TypeError:
        kwargs.pop("dynamic", None)
        model = torch.compile(model, **kwargs)
    except Exception as e:
        log_print(f"torch.compile failed; fallback to eager mode. Error: {repr(e)}", log_fh, is_main)
        return model

    log_print(
        f"Enabled torch.compile: backend={args.compile_backend}, "
        f"mode={args.compile_mode}, dynamic={bool(args.compile_dynamic)}",
        log_fh,
        is_main,
    )

    return model

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    checkpoint_dir,
    scheduler=None,
    scaler=None,
    args=None,
    latest: bool = True,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    if args is not None:
        state["args"] = vars(args) if not isinstance(args, dict) else args

    torch.save(
        state,
        os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth"),
    )

    if latest:
        torch.save(
            state,
            os.path.join(checkpoint_dir, "checkpoint_latest.pth"),
        )


def save_ema(ema, epoch, checkpoint_dir, latest: bool = True):
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": ema.state_dict(),
    }

    torch.save(
        state,
        os.path.join(checkpoint_dir, f"ema_{epoch + 1}.pth"),
    )

    if latest:
        torch.save(
            state,
            os.path.join(checkpoint_dir, "ema_latest.pth"),
        )


def load_resume(
    resume_path,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    ema=None,
    strict_scheduler: bool = True,
    log_fh=None,
    is_main: bool = True,
):
    if resume_path is None:
        return 0

    ckpt = torch.load(resume_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid resume checkpoint: {resume_path}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    log_print(f"Resumed model from {resume_path}", log_fh, is_main)

    if "optimizer_state_dict" not in ckpt:
        raise KeyError("Resume checkpoint has no optimizer_state_dict.")

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log_print("Resumed optimizer state.", log_fh, is_main)

    if scheduler is not None:
        if "scheduler_state_dict" not in ckpt:
            msg = "Resume checkpoint has no scheduler_state_dict."
            if strict_scheduler:
                raise KeyError(msg)
            warnings.warn(msg)
        else:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            log_print("Resumed scheduler state.", log_fh, is_main)

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        log_print("Resumed AMP scaler state.", log_fh, is_main)

    if ema is not None:
        ckpt_dir, ckpt_name = os.path.split(resume_path)

        ema_candidates = []
        if ckpt_name == "checkpoint_latest.pth":
            ema_candidates.append(os.path.join(ckpt_dir, "ema_latest.pth"))

        if ckpt_name.startswith("checkpoint_"):
            ema_candidates.append(
                os.path.join(ckpt_dir, ckpt_name.replace("checkpoint_", "ema_", 1))
            )

        for ema_path in ema_candidates:
            if os.path.exists(ema_path):
                ema_state = torch.load(ema_path, map_location="cpu")["model_state_dict"]
                ema.load_state_dict(ema_state)
                log_print(f"Resumed EMA model from {ema_path}", log_fh, is_main)
                break
        else:
            warnings.warn("EMA checkpoint was not found for resume.")

    start_epoch = int(ckpt["epoch"]) + 1
    log_print(f"Resume start_epoch={start_epoch}", log_fh, is_main)

    return start_epoch


def save_final_models(model, ema, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
    torch.save(ema.state_dict(), os.path.join(checkpoint_dir, "ema_model.pth"))


def load_finetune(
    finetune_path,
    model,
    optimizer=None,
    ema=None,
    lr=None,
    ignore_size_mismatch: bool = False,
    log_fh=None,
    is_main: bool = True,
):
    if finetune_path is None:
        return

    ckpt = torch.load(finetune_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if ignore_size_mismatch:
        load_state_dict_ignore_size_mismatch(model, state_dict)
    else:
        model.load_state_dict(state_dict, strict=False)

    log_print(f"Loaded model from {finetune_path}", log_fh, is_main)

    if optimizer is not None:
        try:
            if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])

                if lr is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = lr

                log_print("Loaded optimizer state.", log_fh, is_main)

        except Exception:
            warnings.warn("Optimizer state dict could not be loaded")

    if ema is not None:
        ckpt_dir, ckpt_name = os.path.split(finetune_path)
        ema_path = os.path.join(ckpt_dir, ckpt_name.replace("checkpoint", "ema"))

        if os.path.exists(ema_path):
            ema_state = torch.load(ema_path, map_location="cpu")["model_state_dict"]
            ema.load_state_dict(ema_state)
            log_print(f"Loaded EMA model from {ema_path}", log_fh, is_main)