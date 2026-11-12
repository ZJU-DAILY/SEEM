from __future__ import annotations
import json
import os
import hashlib
import copy
import random
import numpy as np
from pathlib import Path
import torch
from typing import Sequence
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(str(tmp),str(path))

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');torch.save(obj,tmp);os.replace(str(tmp),str(path))

def read(path):return json.loads(Path(path).read_text())

def bind(folder,protocol,resume):
    folder=Path(folder);p=folder/'protocol.json'
    if p.exists():
        if not resume:raise ValueError('Existing run: use --resume: '+str(folder))
        if read(p)!=protocol:raise ValueError('Resume configuration/source/code changed: '+str(folder))
    elif folder.exists() and any(folder.iterdir()):raise ValueError('Unbound existing output: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True);write(p,protocol)

def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)

def capture():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(kind=n[0],keys=n[1].tolist(),pos=n[2],has_gauss=n[3],cached=n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore(state):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG restore requires CUDA')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false.")
    return device


def multigrain_centered(data: torch.Tensor, periods: Sequence[int]) -> torch.Tensor:

    if data.ndim != 3:
        raise ValueError("data must have shape [items, length, channels].")
    _, length, _ = data.shape
    representations = []
    for period in periods:
        if period < 1 or length % period != 0:
            raise ValueError("Sequence length {} must be divisible by period {}.".format(length, period))
        coarse = data.reshape(data.shape[0], length // period, period, data.shape[2]).mean(dim=2)
        repeated = coarse.repeat_interleave(period, dim=1)
        representations.append(repeated - repeated[:, -1:, :])
    return torch.stack(representations, dim=0)


def normalized_flattened(data: torch.Tensor) -> torch.Tensor:

    if data.ndim != 4:
        raise ValueError("data must have shape [granularity, items, length, channels].")
    flattened = data.flatten(start_dim=2)
    centered = flattened - flattened.mean(dim=2, keepdim=True)
    return F.normalize(centered, p=2, dim=2)


def forbidden_bounds(bank_sources, query_sources, seq_len, pred_len, training):

    if not training:
        return np.zeros(len(query_sources), dtype=np.int64), np.zeros(len(query_sources), dtype=np.int64)
    span = seq_len + pred_len
    lo = np.searchsorted(bank_sources, query_sources - span, side='right')
    hi = np.searchsorted(bank_sources, query_sources + span, side='left')
    return lo, hi


def state_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


