from __future__ import annotations
import functools
import gc
import inspect
from pathlib import Path
import numpy as np
import torch
from utils import read, write


_requested = 128
_rows = 128
_path = None
_events = []


def configure(folder, requested_rows=128):
    global _requested, _rows, _path, _events
    _requested = requested_rows
    _path = Path(folder) / 'execution.json'
    _rows = requested_rows
    _events = []
    if _path.exists():
        record = read(_path)
        if record['requested_rows'] != requested_rows:
            raise ValueError('Requested execution rows changed')
        _rows = record['actual_rows']
        _events = record['oom_events']
    persist()


def persist():
    if _path is not None:
        write(_path, dict(requested_rows=_requested, actual_rows=_rows,
              oom_events=_events, full_A_B_objectives=True,
              optimizer_updates_unchanged=True, fallback_trigger='actual CUDA OOM only'))


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        'cuda' in str(exc).lower() and 'out of memory' in str(exc).lower())


def shrink(where, attempted):
    global _rows
    if attempted <= 1:
        raise RuntimeError('CUDA OOM at one execution row: ' + where)
    _rows = min(_rows, max(1, attempted // 2))
    _events.append(dict(operation=where, attempted_rows=attempted, retry_rows=_rows))
    persist()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('ACTUAL CUDA OOM | %s | execution rows %d -> %d; full objective and Adam updates unchanged' %
          (where, attempted, _rows), flush=True)


def adaptive(fn, rewind_gradients=False):

    signature = inspect.signature(fn)
    if 'chunk' not in signature.parameters:
        raise ValueError('Missing chunk argument')

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        requested = bound.arguments['chunk']
        saved = []
        if rewind_gradients:
            seen = set()
            for value in bound.arguments.values():
                if isinstance(value, torch.nn.Module):
                    for parameter in value.parameters():
                        if id(parameter) not in seen:
                            seen.add(id(parameter))
                            saved.append((parameter, None if parameter.grad is None
                                          else parameter.grad.detach().clone()))
        while True:
            attempted = min(requested, _rows)
            bound.arguments['chunk'] = attempted
            try:
                return fn(*bound.args, **bound.kwargs)
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise


            for parameter, gradient in saved:
                parameter.grad = None if gradient is None else gradient.clone()
            shrink(fn.__name__, attempted)

    return wrapped


def row_blocks(count, work, maximum=None):
    first = 0
    while first < count:
        attempted = min(_rows, maximum or _requested)
        rows = np.arange(first, min(first + attempted, count))
        try:
            block = work(rows)
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
        else:
            yield rows, block
            first += len(rows)
            del block
            continue
        shrink(work.__name__, attempted)


def cache_metrics(cache, chunk=64):
    squared = absolute = 0.
    for first in range(0, len(cache['target']), chunk):
        error = cache['prediction'][first:first+chunk].double() - cache['target'][first:first+chunk].double()
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
    return dict(mse=squared/cache['target'].numel(), mae=absolute/cache['target'].numel())


def channel_std(values, end, chunk=128):

    count = 0
    mean = total = None
    for first in range(0, end, chunk):
        block = values[first:min(first+chunk, end)].double().reshape(-1, values.shape[-1])
        variance, local_mean = torch.var_mean(block, dim=0, unbiased=False)
        size = len(block)
        if mean is None:
            mean = local_mean
            total = variance * size
        else:
            delta = local_mean - mean
            total = total + variance*size + delta.square()*(count*size/(count+size))
            mean = mean + delta*(size/(count+size))
        count += size
    if not count:
        raise ValueError('Empty standard-deviation range')
    return (total/count).clamp_min(0).sqrt()
