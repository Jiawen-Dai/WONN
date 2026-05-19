import argparse
import os
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v

    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def seed_everything(seed: int, deterministic: bool = False):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

def load_state_dict_ignore_size_mismatch(model, state_dict):
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and model_state[k].shape == v.shape
    }

    return model.load_state_dict(filtered, strict=False)