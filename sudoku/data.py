from __future__ import annotations

import os.path as osp
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


BOARD_SIZE = 9
NUM_DIGITS = 9
TRAIN_SIZE = 9000
TEST_SIZE = 1000

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def load_dataset(data_root: str):
    feature_path = osp.join(data_root, "features.pt")
    label_path = osp.join(data_root, "labels.pt")

    if not osp.exists(feature_path):
        raise FileNotFoundError(f"Expected file not found: {feature_path}")
    if not osp.exists(label_path):
        raise FileNotFoundError(f"Expected file not found: {label_path}")

    features = torch.load(feature_path, map_location="cpu")
    labels = torch.load(label_path, map_location="cpu")

    return features, labels


def features_to_tokens(features: torch.Tensor) -> torch.Tensor:
    is_input = features.sum(dim=-1) > 0
    digits = features.argmax(dim=-1).to(torch.long) + 1
    return digits * is_input.to(torch.long)


def labels_to_targets(labels: torch.Tensor) -> torch.Tensor:
    return labels.argmax(dim=-1).to(torch.long)


class SudokuDataset(Dataset):
    def __init__(self, data_root: str = "./data/sudoku", train: bool = True):
        super().__init__()

        features, labels = load_dataset(data_root)

        if features.ndim != 4 or features.shape[1:] != (BOARD_SIZE, BOARD_SIZE, NUM_DIGITS):
            raise ValueError(f"Unexpected feature shape: {tuple(features.shape)}")
        if labels.shape != features.shape:
            raise ValueError(
                f"Feature/label shapes do not match: "
                f"features={tuple(features.shape)}, labels={tuple(labels.shape)}"
            )

        start = 0 if train else TRAIN_SIZE
        end = TRAIN_SIZE if train else TRAIN_SIZE + TEST_SIZE

        if features.shape[0] < end:
            raise ValueError(f"Expected at least {end} boards, got {features.shape[0]}")

        features = features[start:end]
        labels = labels[start:end]

        self.inputs = features_to_tokens(features)
        self.targets = labels_to_targets(labels)
        self.is_input = self.inputs > 0

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Batch:
        return (
            self.inputs[idx].long(),
            self.targets[idx].long(),
            self.is_input[idx].bool(),
        )


def make_worker_init_fn(seed: int, rank: int = 0):
    def _seed_worker(worker_id: int):
        worker_seed = int(seed) + int(rank) * 1000 + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def build_dataloaders(
    data_root: str = "./data/sudoku",
    batch_size: int = 100,
    eval_batch_size: int = 100,
    num_workers: int = 4,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 137,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 4,
):
    trainset = SudokuDataset(data_root=data_root, train=True)
    testset = SudokuDataset(data_root=data_root, train=False)

    train_sampler = (
        DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        if is_ddp
        else None
    )

    test_sampler = (
        DistributedSampler(
            testset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if is_ddp
        else None
    )

    loader_kwargs = dict(
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(seed=seed, rank=rank),
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    testloader = DataLoader(
        testset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    return trainloader, testloader, train_sampler, test_sampler


def move_batch_to_device(batch: Batch, device: torch.device, non_blocking: bool = True) -> Batch:
    inputs, targets, is_input = batch
    return (
        inputs.to(device=device, dtype=torch.long, non_blocking=non_blocking),
        targets.to(device=device, dtype=torch.long, non_blocking=non_blocking),
        is_input.to(device=device, dtype=torch.bool, non_blocking=non_blocking),
    )