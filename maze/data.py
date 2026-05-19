from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


RAW_PAD = 0
RAW_WALL = 1
RAW_FREE = 2
RAW_START = 3
RAW_GOAL = 4
RAW_PATH = 5


MAZE_IN_WALL = 0
MAZE_IN_FREE = 1
MAZE_IN_START = 2
MAZE_IN_GOAL = 3
NUM_INPUT_TOKENS = 4


MAZE_OUT_WALL = 0
MAZE_OUT_FREE = 1
MAZE_OUT_START = 2
MAZE_OUT_GOAL = 3
MAZE_OUT_PATH = 4
NUM_OUTPUT_TOKENS = 5


IGNORE_INDEX = -100
MAZE_SIZE = 30
MAZE_CELLS = MAZE_SIZE * MAZE_SIZE


class MazeDataset(Dataset):
    """Load Maze-Hard arrays and remap them to 4-class input / 5-class output."""

    def __init__(self, root: str, split: str = "train"):
        super().__init__()

        split_dir = os.path.join(root, split)
        input_path = os.path.join(split_dir, "all__inputs.npy")
        label_path = os.path.join(split_dir, "all__labels.npy")

        self.inputs_raw = np.load(input_path)
        self.labels_raw = np.load(label_path)

        if self.inputs_raw.shape != self.labels_raw.shape:
            raise ValueError("Maze input/label shapes do not match.")

        if self.inputs_raw.ndim != 2 or self.inputs_raw.shape[1] != MAZE_CELLS:
            raise ValueError(f"Expected [N, 900] arrays, got {self.inputs_raw.shape}")

        self.inputs = self._remap_inputs(self.inputs_raw)
        self.targets = self._remap_targets(self.labels_raw)

    @staticmethod
    def _remap_inputs(raw: np.ndarray) -> np.ndarray:
        out = np.full_like(raw, fill_value=MAZE_IN_FREE, dtype=np.int64)

        out[raw == RAW_WALL] = MAZE_IN_WALL
        out[raw == RAW_FREE] = MAZE_IN_FREE
        out[raw == RAW_START] = MAZE_IN_START
        out[raw == RAW_GOAL] = MAZE_IN_GOAL

        out[raw == RAW_PATH] = MAZE_IN_FREE
        out[raw == RAW_PAD] = MAZE_IN_FREE

        return out.reshape(-1, MAZE_SIZE, MAZE_SIZE)

    @staticmethod
    def _remap_targets(raw: np.ndarray) -> np.ndarray:
        out = np.full_like(raw, fill_value=IGNORE_INDEX, dtype=np.int64)

        out[raw == RAW_WALL] = MAZE_OUT_WALL
        out[raw == RAW_FREE] = MAZE_OUT_FREE
        out[raw == RAW_START] = MAZE_OUT_START
        out[raw == RAW_GOAL] = MAZE_OUT_GOAL
        out[raw == RAW_PATH] = MAZE_OUT_PATH

        return out.reshape(-1, MAZE_SIZE, MAZE_SIZE)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.inputs[idx]).long()
        y = torch.from_numpy(self.targets[idx]).long()
        return x, y


def build_datasets(data_root: str):
    trainset = MazeDataset(data_root, split="train")
    testset = MazeDataset(data_root, split="test")
    return trainset, testset


def build_dataloaders(
    data_root: str,
    batch_size: int = 64,
    test_batch_size: int = 100,
    num_workers: int = 4,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    worker_init_fn=None,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 4,
):
    trainset, testset = build_datasets(data_root)

    train_sampler = (
        DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
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
        worker_init_fn=worker_init_fn,
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
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    return trainloader, testloader, train_sampler, test_sampler