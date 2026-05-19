import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_STATS = {
    "imagenet100": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 100,
    },
    "imagenet1k": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 1000,
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


def train_transform(data: str, img_size: int = 224):
    mean = IMAGENET_STATS[data]["mean"]
    std = IMAGENET_STATS[data]["std"]

    if data == "imagenet100":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if data == "imagenet1k":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25),
        ])

    raise ValueError(f"data must be one of {list(IMAGENET_STATS.keys())}, but got {data!r}.")


def eval_transform(data: str, img_size: int = 224, eval_resize: int = 256):
    mean = IMAGENET_STATS[data]["mean"]
    std = IMAGENET_STATS[data]["std"]

    return transforms.Compose([
        transforms.Resize(
            eval_resize,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def rand_bbox(size, lam):
    height = size[-2]
    width = size[-1]

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    return x1, y1, x2, y2


def apply_cutmix(inputs, labels, alpha: float = 1.0, prob: float = 1.0):
    if alpha <= 0.0 or prob <= 0.0 or torch.rand(1).item() >= prob:
        return inputs, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(inputs.size(0), device=inputs.device)

    labels_a = labels
    labels_b = labels[rand_index]

    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[rand_index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (inputs.size(-1) * inputs.size(-2)))
    return mixed_inputs, labels_a, labels_b, lam


def build_datasets(
    data: str,
    data_root: str,
    img_size: int = 224,
    eval_resize: int = 256,
):
    data = str(data).lower()
    if data not in IMAGENET_STATS:
        raise ValueError(f"data must be one of {list(IMAGENET_STATS.keys())}, but got {data!r}.")

    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_root, "train"),
        transform=train_transform(data, img_size=img_size),
    )

    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_root, "val"),
        transform=eval_transform(data, img_size=img_size, eval_resize=eval_resize),
    )

    expected_classes = IMAGENET_STATS[data]["num_classes"]
    if len(trainset.classes) != expected_classes or len(testset.classes) != expected_classes:
        raise ValueError(
            f"{data} expects {expected_classes} classes, "
            f"but got train={len(trainset.classes)}, val={len(testset.classes)}."
        )

    return trainset, testset


def build_dataloaders(
    data: str,
    data_root: str,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    workers: int = 8,
    img_size: int = 224,
    eval_resize: int = 256,
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
        img_size=img_size,
        eval_resize=eval_resize,
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
