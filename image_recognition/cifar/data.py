import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


CIFAR_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
    },
}

class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas: int = 1, rank: int = 0):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        dataset_size = len(dataset)
        per_rank = int(np.ceil(dataset_size / self.num_replicas))

        start = self.rank * per_rank
        end = min(start + per_rank, dataset_size)
        self.indices = list(range(start, end))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_worker_init_fn(base_seed: int, rank: int):
    def _seed_worker(worker_id: int):
        worker_seed = int(base_seed) + int(rank) * 1000 + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def train_transform(data: str, strong_aug: bool = True):
    mean = CIFAR_STATS[data]["mean"]
    std = CIFAR_STATS[data]["std"]

    if strong_aug:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def eval_transform(data: str):
    mean = CIFAR_STATS[data]["mean"]
    std = CIFAR_STATS[data]["std"]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_datasets(data: str, data_root: str, strong_aug: bool = True, download: bool = False):
    data = str(data).lower()
    if data not in CIFAR_STATS:
        raise ValueError(f"data must be one of {list(CIFAR_STATS.keys())}, but got {data!r}.")

    dataset_cls = torchvision.datasets.CIFAR10 if data == "cifar10" else torchvision.datasets.CIFAR100

    trainset = dataset_cls(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform(data, strong_aug=strong_aug),
    )

    testset = dataset_cls(
        root=data_root,
        train=False,
        download=download,
        transform=eval_transform(data),
    )

    return trainset, testset


def build_dataloaders(
    data: str,
    data_root: str,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    workers: int = 8,
    strong_aug: bool = True,
    download: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    seed: int = 137,
    is_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    trainset, testset = build_datasets(
        data=data,
        data_root=data_root,
        strong_aug=strong_aug,
        download=download,
    )

    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    ) if is_ddp else None

    test_sampler = DistributedEvalSampler(
        testset,
        num_replicas=world_size,
        rank=rank,
    ) if is_ddp else None

    generator = torch.Generator()
    generator.manual_seed(seed + rank)

    loader_kwargs = dict(
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
        worker_init_fn=make_worker_init_fn(seed, rank),
        generator=generator,
    )

    if workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
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