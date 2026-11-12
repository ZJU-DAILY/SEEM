from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from functools import lru_cache
from utils import cpu_copy
from networks import Content, Writer, load_content
import numpy as np
import os
import hashlib
import time
from pathlib import Path
from utils import read, write, save
from data import digest


@lru_cache(maxsize=16)
def coarse_basis(horizon):
    if horizon < 4:
        raise ValueError('At least four forecast steps required')
    basis=F.interpolate(torch.eye(4,dtype=torch.float64).unsqueeze(1),size=horizon,mode='linear',align_corners=True).squeeze(1).T
    return torch.linalg.qr(basis,mode='reduced')[0]

def components(coeff,horizon):
    if coeff.shape[-2] != 21:
        raise ValueError('Expected 1+4+16 temporal coefficients per channel')
    overall=coeff[:,:1].expand(-1,horizon,-1)
    coarse=F.interpolate(coeff[:,1:5].transpose(1,2),size=horizon,mode='linear',align_corners=True).transpose(1,2)
    coarse=coarse-coarse.mean(1,keepdim=True)
    fine=F.interpolate(coeff[:,5:].transpose(1,2),size=horizon,mode='linear',align_corners=True).transpose(1,2)
    q=coarse_basis(horizon).to(device=coeff.device,dtype=coeff.dtype)

    fine=fine-torch.einsum('hr,grc->ghc',q,torch.einsum('hr,ghc->grc',q,fine))
    return overall,coarse,fine

def effect(delta, projection, fusion):
    return F.linear(F.linear(delta.transpose(-1, -2), projection), fusion).transpose(-1, -2)

def normalized(x):
    x = x.detach().double()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)

def temporal_basis(horizon, device='cpu'):
    identity = torch.eye(21, dtype=torch.float64, device=device)[:, :, None]
    return sum(components(identity, horizon)).squeeze(-1).T.contiguous()

def decode(coefficients, basis, scale):
    pre = torch.einsum('hl,...lc->...hc', basis, coefficients.double())
    return .25 * scale * pre.tanh()

def pooled(x, blocks=4):
    horizon, channels = x.shape[-2:]
    if horizon % blocks: raise ValueError('Expected equal temporal pooling blocks')
    return x.reshape(*x.shape[:-2], blocks, horizon // blocks, channels).mean(-2).flatten(-2)

def observable_features(query, group, weights, value_summary, m0_summary):

    q = query[:, None].expand_as(group)
    pred = m0_summary[:, None].expand(*group.shape[:2], m0_summary.shape[-1])
    return torch.cat((q, group, q * group, (q - group).abs(), weights[..., None],
                      value_summary, pred), -1).float()

def gradient_norm(module):
    return math.sqrt(sum(float(p.grad.double().square().sum()) for p in module.parameters() if p.grad is not None))


ROLES = ('train', 'validation', 'test', 'A', 'B')


def fit_query_map(query, a_end, dimension=32):

    x = query[:a_end].detach().cpu().double()
    if not 0 < dimension <= x.shape[1]: raise ValueError('Invalid fixed query dimension')
    center = x.mean(0); z = x - center
    eigenvalues, vectors = torch.linalg.eigh(z.T @ z / len(z))
    axes = vectors[:, -dimension:].flip(1).contiguous()
    maxima = axes.abs().argmax(0)
    signs = axes[maxima, torch.arange(dimension)].sign(); signs[signs == 0] = 1
    axes = axes * signs
    deviation = (z @ axes).square().mean(0).sqrt().clamp_min(1e-6)
    return dict(center=center, axes=axes, deviation=deviation, dimension=dimension,
                fit_rows=[0, a_end], eigenvalues=eigenvalues[-dimension:].flip(0),
                definition='A-only centered PCA coordinates / A population std, floor1e-6; no query L2 renormalization after PCA')


def query_coordinates(query, mapping, device=None):
    x = query.to(device=device).double()
    return (x - mapping['center'].to(x)) @ mapping['axes'].to(x) / mapping['deviation'].to(x)


def initial_applicability(groups, dimension, device):
    state = torch.zeros(groups, dimension + 1, dtype=torch.float64, device=device)
    state[:, -1] = math.log(99.)
    return state


def gates(cache, rows, state):
    ids = cache['ids'][rows].to(state.device)
    query = cache['psi'][rows].to(state)
    selected = state[ids]
    logits = selected[..., -1] + (query[:, None] * selected[..., :-1]).sum(-1) / math.sqrt(query.shape[-1])
    return logits.sigmoid()


def prediction(content, cache, rows, state, basis, scale, projection, fusion, per_read=False):
    ids = cache['ids'][rows].to(scale.device)
    coefficients = content(cache['observables'][rows].to(scale.device), ids)
    raw = decode(coefficients, basis, scale)
    alpha = gates(cache, rows, state)
    delta = raw * alpha[..., None, None]
    weights = cache['weights'][rows].to(scale)
    response = effect(raw, projection, fusion) * weights[..., None, None]
    result = cache['prediction'][rows].to(scale) + (response * alpha[..., None, None]).sum(1)
    return result, coefficients, delta, alpha, response if per_read else None


def fit_normalization(cache, a_end, chunk=64):
    total = square = None; count = 0
    for lo in range(0, a_end, chunk):
        x = cache['observables'][lo:min(lo + chunk, a_end)].double().flatten(0, 1)
        if total is None: total = x.sum(0); square = x.square().sum(0)
        else: total += x.sum(0); square += x.square().sum(0)
        count += len(x)
    center = total / count
    return center.float(), (square / count - center.square()).clamp_min(0).sqrt().clamp_min(1e-6).float()


def A_gradient(content, cache, state, basis, scale, projection, fusion, a_end, chunk=64):
    if state.requires_grad: raise ValueError('Applicability must be fixed during A')
    total = 0.
    for lo in range(0, a_end, chunk):
        rows = slice(lo, min(lo + chunk, a_end))
        pred = prediction(content, cache, rows, state, basis, scale, projection, fusion)[0]
        loss = (pred - cache['target'][rows].to(scale)).square().sum() / (a_end * pred.shape[-2] * pred.shape[-1])
        if not torch.isfinite(loss): raise ValueError('Nonfinite A objective')
        loss.backward(); total += float(loss.detach())
    return dict(objective_before_update=total, A_gradient_norm=gradient_norm(content))


@torch.no_grad()
def freeze_actions(content, caches, state, basis, scale, projection, fusion, chunk=64, compact=False):

    fixed = {}
    for sp, cache in caches.items():
        responses = []; gram = []; cross = []; base_error = []; value_mean_sq = []; raw_bound = []
        for lo in range(0, len(cache['target']), chunk):
            rows = slice(lo, min(lo + chunk, len(cache['target'])))
            _, coefficients, _, _, response = prediction(content, cache, rows, state, basis, scale, projection, fusion, True)
            raw = decode(coefficients, basis, scale)
            value_mean_sq.append(raw.square().mean((-2, -1)).cpu())
            raw_bound.append((raw / (.25 * scale)).abs().flatten(-2).max(-1).values.cpu())
            flat = response.flatten(-2)
            error = cache['prediction'][rows].to(scale) - cache['target'][rows].to(scale)
            gram.append((flat @ flat.transpose(-1, -2) / error[0].numel()).cpu())
            cross.append((response * error[:, None]).mean((-2, -1)).cpu())
            base_error.append(error.square().mean((-2, -1)).cpu())
            responses.append(coefficients.cpu() if compact else response.cpu())
            if lo % 2048 == 0: print('Frozen post-A actions | %s | %d/%d' % (sp, rows.stop, len(cache['target'])), flush=True)
        fixed[sp] = dict(gram=torch.cat(gram), cross=torch.cat(cross),
                         base_error=torch.cat(base_error), value_mean_sq=torch.cat(value_mean_sq),
                         raw_bound=torch.cat(raw_bound))
        if compact:
            fixed[sp].update(coefficients=torch.cat(responses), basis=basis.cpu(), scale=scale.cpu(),
                             projection=projection.cpu(), fusion=fusion.cpu())
        else:
            fixed[sp]['response']=torch.cat(responses)
    return fixed


def range_mse(cache, fixed, state, first, last, chunk=64, backward=False, weight=1.):

    total = 0.; n = last - first
    if n < 1: raise ValueError('Empty MSE range')
    with torch.set_grad_enabled(backward):
        for lo in range(first, last, chunk):
            rows = slice(lo, min(lo + chunk, last)); alpha = gates(cache, rows, state)
            gram = fixed['gram'][rows].to(state); cross = fixed['cross'][rows].to(state)
            value = fixed['base_error'][rows].to(state) + 2 * (cross * alpha).sum(-1) +\
                (alpha * (gram @ alpha[..., None]).squeeze(-1)).sum(-1)
            loss = value.sum() / n
            if not torch.isfinite(loss): raise ValueError('Nonfinite applicability objective')
            if backward: (weight * loss).backward()
            total += float(loss.detach())
    return total


def state_gradient(cache, fixed, state, first, last, chunk=64):
    leaf = state.detach().clone().requires_grad_(True)
    loss = range_mse(cache, fixed, leaf, first, last, chunk, True)
    return leaf.grad.detach(), loss


def writer_candidate(writer, feedback, anchor):
    if feedback['input'].requires_grad or anchor.requires_grad: raise ValueError('Feedback and write anchor must be detached')
    update = writer(feedback['input'].to(anchor.device)).double()
    candidate = anchor + update * feedback['write_mask'].to(anchor.device)[:, None]
    if not torch.isfinite(candidate).all(): raise ValueError('Nonfinite Writer candidate state')
    return candidate


def writer_gradient(writer, feedback, anchor, cache, fixed, a_end, b_start, limit, chunk=64):
    candidate = writer_candidate(writer, feedback, anchor)
    leaf = candidate.detach().requires_grad_(True)
    a_mse = range_mse(cache, fixed, leaf, 0, a_end, chunk)
    b_mse = range_mse(cache, fixed, leaf, b_start, len(cache['target']), chunk, True)
    excess = max(a_mse - limit, 0.)
    if excess > 0: range_mse(cache, fixed, leaf, 0, a_end, chunk, True)
    candidate.backward(leaf.grad)
    return dict(objective_before_update=b_mse + excess, B_mse_before_update=b_mse,
                A_mse_before_update=a_mse, A_excess_before_update=excess,
                Writer_gradient_norm=gradient_norm(writer), second_order=False)


@torch.no_grad()
def observe(content, caches, state, basis, scale, projection, fusion, a_end, b_start, chunk=64, fixed=None):
    metrics = {}; diagnostics = {}
    for sp, cache in caches.items():
        se = ae = change_sq = alpha_sum = value_sq = 0.; events = 0; near0 = near1 = 0; bound = 0.
        region = {name: [0., 0., 0] for name in ('A', 'B')} if sp == 'train' else {}
        for lo in range(0, len(cache['target']), chunk):
            rows = slice(lo, min(lo + chunk, len(cache['target'])))
            if fixed is None:
                pred, _, delta, alpha, _ = prediction(content, cache, rows, state, basis, scale, projection, fusion)
                bound = max(bound, float((delta / (.25 * scale)).abs().max()))
                value_sq += float(delta.square().mean((-2, -1)).sum())
            else:
                alpha = gates(cache, rows, state)
                pred = cache['prediction'][rows].to(scale) + (action_response(fixed[sp], cache, rows, scale) * alpha[..., None, None]).sum(1)
                value_sq += float((fixed[sp]['value_mean_sq'][rows].to(scale) * alpha.square()).sum())
                bound = max(bound, float((fixed[sp]['raw_bound'][rows].to(scale) * alpha).max()))
            error = pred - cache['target'][rows].to(scale)
            se += float(error.square().sum()); ae += float(error.abs().sum())
            change_sq += float((pred - cache['prediction'][rows].to(scale)).square().sum())
            alpha_sum += float(alpha.sum()); events += alpha.numel()
            near0 += int((alpha < .01).sum()); near1 += int((alpha > .99).sum())
            for name, first, last in [('A', 0, a_end), ('B', b_start, len(cache['target']))] if sp == 'train' else []:
                left = max(lo, first); right = min(rows.stop, last)
                if right > left:
                    e = error[left - lo:right - lo]
                    region[name][0] += float(e.square().sum()); region[name][1] += float(e.abs().sum()); region[name][2] += e.numel()
        size = cache['target'].numel()
        metrics[sp] = dict(mse=se / size, mae=ae / size)
        diagnostics[sp] = dict(response_rms=math.sqrt(change_sq / size), gate_mean=alpha_sum / events,
            gate_below01_fraction=near0 / events, gate_above99_fraction=near1 / events,
            effective_bound_max=bound, value_delta_rms=math.sqrt(value_sq / events))
        for name, (s, a, count) in region.items(): metrics[name] = dict(mse=s / count, mae=a / count)
    if any(not math.isfinite(value) for m in metrics.values() for value in m.values()): raise ValueError('Nonfinite observed metrics')
    return metrics, diagnostics


def action_response(fixed, cache, rows, reference):

    if 'response' in fixed:
        return fixed['response'][rows].to(reference)
    raw=decode(fixed['coefficients'][rows].to(reference.device), fixed['basis'].to(reference), fixed['scale'].to(reference))
    return effect(raw,fixed['projection'].to(reference),fixed['fusion'].to(reference))*cache['weights'][rows].to(reference)[...,None,None]


def independent_events(ids, starts, first, last, groups, window=192):

    selected = [[] for _ in range(groups)]
    if window < 1: raise ValueError('Invalid source window length')
    flat = ids[first:last].reshape(-1).cpu().numpy()
    order = np.argsort(flat, kind='stable'); ordered = flat[order]
    for g in range(groups):
        left = np.searchsorted(ordered, g, side='left'); right = np.searchsorted(ordered, g, side='right')
        next_legal = -float('inf')
        for event in order[left:right]:
            row = first + int(event) // ids.shape[1]; slot = int(event) % ids.shape[1]
            if int(starts[row]) >= next_legal:
                selected[g].append((row, slot)); next_legal = int(starts[row]) + window
    return selected


@torch.no_grad()
def side_evidence(cache, fixed, state, starts, first, last, chunk=64, window=192):
    groups, width = state.shape; dimension = width - 1
    utilities = []; unit_sq = torch.zeros(groups, dtype=torch.float64)
    raw_count = torch.zeros(groups, dtype=torch.float64); mass = torch.zeros_like(raw_count)
    for lo in range(first, last, chunk):
        rows = slice(lo, min(lo + chunk, last)); ids = cache['ids'][rows]
        alpha = gates(cache, rows, state)
        unit = action_response(fixed, cache, rows, state)
        contribution = unit * alpha[..., None, None]
        prediction = cache['prediction'][rows].to(state) + contribution.sum(1)
        error = prediction - cache['target'][rows].to(state)

        utility = (contribution.square() - 2 * error[:, None] * contribution).mean((-2, -1))
        utilities.append(utility.cpu())
        flat_ids = ids.flatten().cpu()
        raw_count.index_add_(0, flat_ids, torch.ones(len(flat_ids), dtype=torch.float64))
        mass.index_add_(0, flat_ids, cache['weights'][rows].flatten().double().cpu())
        unit_sq.index_add_(0, flat_ids, unit.square().mean((-2, -1)).flatten().cpu())
    utilities = torch.cat(utilities)
    independent = independent_events(cache['ids'], starts, first, last, groups, window)
    positive = torch.zeros(groups, dimension, dtype=torch.float64); negative = torch.zeros_like(positive)
    independent_count = torch.tensor([len(x) for x in independent], dtype=torch.float64)
    pos_count = torch.zeros(groups, dtype=torch.float64); neg_count = torch.zeros_like(pos_count)
    pos_mass = torch.zeros_like(pos_count); neg_mass = torch.zeros_like(pos_count)
    for g, events in enumerate(independent):
        for row, slot in events:
            utility = float(utilities[row - first, slot]); psi = cache['psi'][row].double().cpu()
            if utility > 1e-10:
                positive[g] += utility * psi; pos_mass[g] += utility; pos_count[g] += 1
            elif utility < -1e-10:
                negative[g] += -utility * psi; neg_mass[g] -= utility; neg_count[g] += 1
    positive /= pos_mass.clamp_min(1e-30)[:, None]; negative /= neg_mass.clamp_min(1e-30)[:, None]
    utility_scale = max(float(utilities.square().mean().sqrt()), 1e-10)
    unit_rms = (unit_sq / raw_count.clamp_min(1)).sqrt()
    response_scale = max(float((unit_sq.sum() / raw_count.sum().clamp_min(1)).sqrt()), 1e-12)
    statistics = torch.stack((raw_count.log1p(), independent_count.log1p(), mass.log1p(), pos_count.log1p(),
        neg_count.log1p(), pos_mass / utility_scale, neg_mass / utility_scale, unit_rms / response_scale), -1)
    return dict(positive=positive, negative=negative, statistics=statistics, utilities=utilities,
        first=first, last=last, raw_count=raw_count, independent_count=independent_count,
        positive_count=pos_count, negative_count=neg_count, positive_mass=pos_mass, negative_mass=neg_mass,
        unit_response_rms=unit_rms, utility_scale=utility_scale, response_scale=response_scale,
        independent_events=[torch.tensor(x, dtype=torch.long).reshape(-1, 2) for x in independent],
        utility_definition='MSE(full prediction minus current gated correction)-MSE(full prediction); other19 and original Value fixed')


def collect_feedback(cache, fixed, state, starts, a_end, b_start, chunk=64, window=192):
    if state.requires_grad: raise ValueError('Actual write anchor must be detached')
    gradient_A, a_mse = state_gradient(cache, fixed, state, 0, a_end, chunk)
    gradient_B, b_mse = state_gradient(cache, fixed, state, b_start, len(cache['target']), chunk)
    A = side_evidence(cache, fixed, state, starts, 0, a_end, chunk, window)
    B = side_evidence(cache, fixed, state, starts, b_start, len(cache['target']), chunk, window)
    gradients = []; scales = {}
    for name, gradient, side in [('A', gradient_A, A), ('B', gradient_B, B)]:
        supported = side['raw_count'].to(gradient.device) > 0
        rms = max(float(gradient[supported].square().mean().sqrt()) if bool(supported.any()) else 0., 1e-12)
        scales[name] = rms; gradients.append(gradient.detach().cpu() / rms)

    write_mask = (B['independent_count'] >= 2) & (B['unit_response_rms'] > 1e-14)
    inputs = torch.cat((state.detach().cpu(), *gradients, A['positive'], A['negative'], B['positive'], B['negative'],
                        A['statistics'], B['statistics']), -1).float().detach()
    if not torch.isfinite(inputs).all(): raise ValueError('Nonfinite endpoint feedback')
    return dict(input=inputs, write_mask=write_mask, gradient_A=gradient_A.cpu(), gradient_B=gradient_B.cpu(),
        gradient_RMS=scales, A=A, B=B, anchor=state.detach().cpu().clone(), A_mse=a_mse, B_mse=b_mse,
        input_dimension=inputs.shape[-1], write_groups=int(write_mask.sum()), window=window,
        fixed_endpoint=True, feedback_detached=True, second_order=False,
        statistic_columns=['log1p raw reads', 'log1p independent reads', 'log1p read weight mass',
                           'log1p positive support', 'log1p negative support', 'normalized positive utility mass',
                           'normalized negative utility mass', 'normalized ungated weighted response RMS'])


def feedback_summary(feedback):
    return dict(input_dimension=feedback['input_dimension'], write_groups=feedback['write_groups'],
        gradient_RMS=feedback['gradient_RMS'], A_mse=feedback['A_mse'], B_mse=feedback['B_mse'],
        B_missing_groups=int((feedback['B']['raw_count'] == 0).sum()),
        B_less_than_two_independent=int((feedback['B']['independent_count'] < 2).sum()),
        positive_context_groups={s: int((feedback[s]['positive_count'] > 0).sum()) for s in ('A', 'B')},
        negative_context_groups={s: int((feedback[s]['negative_count'] > 0).sum()) for s in ('A', 'B')},
        feedback_detached=True, utility_deadzone=1e-10,
        independent_support='earliest full%d-step nonoverlap per memory, selected without utility' % feedback['window'],
        original_Value_retained=True, marginal_utilities_not_additive=True)


def print_scores(label, metrics):
    print(label + ' | ' + ' | '.join('%s MSE %.10f MAE %.10f' %
        (name, metrics[name]['mse'], metrics[name]['mae']) for name in ROLES if name != 'test'), flush=True)


def tensor_hash(tensor):
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256((str(x.dtype) + str(tuple(x.shape))).encode() + x.numpy().tobytes()).hexdigest()


def module_hash(module):
    return hashlib.sha256(''.join(name + tensor_hash(x) for name, x in module.state_dict().items()).encode()).hexdigest()


def checked_step(optimizer, module):
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
        raise ValueError('Nonfinite new-module gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in module.parameters()): raise ValueError('Nonfinite new-module weight')


def rng_state():
    return dict(cpu=torch.get_rng_state().cpu(), cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available(): raise ValueError('Cannot restore CUDA RNG without CUDA')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def sealed_object(folder, name, binding, create):
    path = folder / (name + '.pt'); seal = folder / (name + '.json')
    if seal.exists():
        record = read(seal)
        if record['binding'] != binding or digest(path) != record['sha256']: raise ValueError('Changed sealed ' + name)
        return torch.load(path, map_location='cpu', weights_only=True)
    value = create(); save(path, value)
    write(seal, dict(binding=binding, sha256=digest(path)))
    return value


def store_endpoint(path, value):
    save(path, value)
    write(path.with_suffix('.seal.json'), dict(binding=value['binding'], sha256=digest(path)))


def read_endpoint(path, binding):
    path = Path(path); seal = read(path.with_suffix('.seal.json'))
    if seal['binding'] != binding or seal['sha256'] != digest(path): raise ValueError('Changed endpoint ' + str(path))
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['binding'] != binding: raise ValueError('Changed endpoint binding')
    return value


def solve(folder, caches, scale, projection, fusion, binding, starts, groups=2048,
          A_steps=100, W_steps=100, a_end=6337, b_start=6528, content_dim=32,
          hidden=64, chunk=64, synthetic=False, window=192, stop_after=None, checkpoint_interval=10, rounds=3, A_lr=.001, W_lr=.001):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    dimension = caches['train']['psi'].shape[-1]; total = rounds * (A_steps + W_steps)
    if rounds < 1 or (not synthetic and rounds != 5): raise ValueError('Expected five fresh rounds')
    if min(A_lr,W_lr)<=0: raise ValueError('Positive A/Writer learning rates required')
    horizon = projection.shape[0]
    if not synthetic:
        from horizons import role_split, expected_counts
        roles = role_split(expected_counts(horizon)['train'], horizon)
        from resources import allowed
        if not allowed(A_steps,W_steps,A_lr,W_lr,horizon) or (rounds,groups,dimension,content_dim,hidden,chunk) != (5,2048,32,32,64,64):
            raise ValueError('Unsupported sweep case or model dimensions')
        if (a_end,b_start,len(caches['train']['target']),window) != (roles['A'][1],roles['B'][0],roles['B'][1],roles['window']):
            raise ValueError('Horizon-specific chronological roles required')
    if not 0 < a_end < b_start < len(caches['train']['target']) or min(A_steps, W_steps) < 1:
        raise ValueError('Invalid A/purge/B or phase budget')
    settings = dict(rounds=rounds, A_steps=A_steps, Writer_steps=W_steps, total_updates=total,
        A_lr=A_lr, Writer_lr=W_lr, A=[0, a_end], B=[b_start, len(caches['train']['target'])],
        purge=[a_end, b_start], groups=groups, content_dim=content_dim, query_dim=dimension,
        horizon=horizon, hidden=hidden, chunk=chunk, seed=int(os.environ['SEEM_RUN_SEED']), window=window, second_order=False,
        optimizer='two continuous Adam A_lr%g Writer_lr%g betas.9/.999 eps1e-8 weight_decay0; no resets' % (A_lr,W_lr),
        initialization='zero A final head/codes; p0=0,b0=log99 ->alpha.99; exact original M0',
        content='shared observable+content-code MLP ->21xchannels; original cumulative .25 tanh1+4+16',
        matching='fixed A-only PCA32 coordinates, scaled inner product /sqrt32; only p,b evolve',
        writer='shared MLP ->delta(p,b), zero initial final head; detached fixed endpoint feedback',
        write='all trials use same actual pre-Writer p,b; commit actual final trial ONCE after each Writer phase',
        write_support='>=2 earliest full%d-step nonoverlap B readers and nonzero ungated correction response' % window,
        protection_fraction=.1, protection_weight=1., state_step_cap=None, extra_regularizer=False,
        objective='Writer: B full-prediction MSE +relu(A MSE-after_A-.1*max(round-start A-after_A,0))',
        feedback='current state,A/B gradients,A+/A-/B+/B- query contexts,16 strength/coverage statistics',
        selection='Validation earliest best ONLY among formal round endpoints; abs1e-12 ties',
        no_rollback=True, no_early_stop=True, historical_evolved_state_import=False,
        B_is_adaptation_supervision=True, B_is_system_OOF=False)
    lock = dict(binding=binding, settings=settings)
    if (folder / 'settings.json').exists() and read(folder / 'settings.json') != lock:
        raise ValueError('Changed solver settings/binding')
    write(folder / 'settings.json', lock)
    torch.manual_seed(int(os.environ['SEEM_RUN_SEED']))
    center, deviation = fit_normalization(caches['train'], a_end, chunk)
    content = Content(center.to(scale.device), deviation.to(scale.device), groups, len(scale), content_dim, hidden)
    state = initial_applicability(groups, dimension, scale.device)
    writer = Writer(7 * dimension + 19, dimension + 1, hidden, scale.device)
    optimizer_A = torch.optim.Adam(content.parameters(), lr=A_lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    optimizer_W = torch.optim.Adam(writer.parameters(), lr=W_lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    basis = temporal_basis(projection.shape[0], scale.device)
    done = 0; history = []; states = {}; anchors = {}; commits = 0; best = None
    pointer = folder / 'resume.json'
    if pointer.exists():
        record = read(pointer); path = folder / record['path']
        if digest(path) != record['sha256']: raise ValueError('Changed resume snapshot')
        snapshot = torch.load(path, map_location='cpu', weights_only=True)
        if snapshot['binding'] != binding or snapshot['settings'] != settings: raise ValueError('Changed resume binding')
        content.load_state_dict(snapshot['content']); writer.load_state_dict(snapshot['writer'])
        optimizer_A.load_state_dict(snapshot['optimizer_A']); optimizer_W.load_state_dict(snapshot['optimizer_W'])
        state = snapshot['applicability'].to(scale); done = snapshot['completed_steps']
        history = snapshot['history']; states = snapshot['states']; anchors = snapshot['anchors']
        commits = snapshot['formal_commits']; best = snapshot['best']; restore_rng(snapshot['rng'])
        if len(history) != done + 1: raise ValueError('Incomplete resume history')
        print('Resume one-write trajectory | actual update %d/%d | formal writes %d | A/Writer/Adam/RNG retained' %
              (done, total, commits), flush=True)

    def full_state(number, phase, local, metrics, diagnostics, proposed=None):
        return dict(binding=binding, settings=settings, round=number, phase=phase, phase_step=local,
            global_step=done, content=content.artifact(), writer=writer.artifact(),
            applicability=cpu_copy(state if proposed is None else proposed),
            optimizer_A=cpu_copy(optimizer_A.state_dict()), optimizer_W=cpu_copy(optimizer_W.state_dict()),
            rng=rng_state(), metrics=cpu_copy(metrics), diagnostics=cpu_copy(diagnostics))

    def checkpoint():
        name = 'resume_%04d.pt' % done
        save(folder / name, dict(binding=binding, settings=settings, completed_steps=done,
            content=cpu_copy(content.state_dict()), writer=cpu_copy(writer.state_dict()), applicability=state.cpu(),
            optimizer_A=cpu_copy(optimizer_A.state_dict()), optimizer_W=cpu_copy(optimizer_W.state_dict()), rng=rng_state(),
            history=history, states=states, anchors=anchors, formal_commits=commits, best=best))
        write(pointer, dict(path=name, sha256=digest(folder / name), completed_steps=done, formal_commits=commits))
        for path in sorted(folder.glob('resume_*.pt'))[:-2]: path.unlink()
        write(folder / 'history.json', history)

    if not history:
        metrics, physical = observe(content, caches, state, basis, scale, projection, fusion, a_end, b_start, chunk)
        diagnostics = dict(physical=physical, formal_write=False)
        states['M0'] = dict(metrics=metrics, diagnostics=diagnostics, global_step=0)
        save(folder / 'initial.pt', full_state(0, 'initial', 0, metrics, diagnostics))
        history.append(dict(global_step=0, round=0, phase='initial', phase_step=0, metrics=metrics, diagnostics=diagnostics))
        print_scores('M0', metrics); checkpoint()
    for number in range(1, rounds + 1):
        start = (number - 1) * (A_steps + W_steps); A_end = start + A_steps; W_end = A_end + W_steps
        if done >= W_end: continue
        rd = folder / ('round%d' % number); rd.mkdir(exist_ok=True)
        key = str(number)
        if key not in anchors: anchors[key] = dict(round_start_A=history[-1]['metrics']['A']['mse'])
        content.train().requires_grad_(True); writer.eval().requires_grad_(False)
        optimizer_W.zero_grad(set_to_none=True)
        for local in range(max(done - start, 0) + 1, A_steps + 1):
            tic = time.monotonic(); optimizer_A.zero_grad(set_to_none=True)
            diagnostics = A_gradient(content, caches['train'], state, basis, scale, projection, fusion, a_end, chunk)
            checked_step(optimizer_A, content); optimizer_A.zero_grad(set_to_none=True)
            optimize = time.monotonic() - tic; tic = time.monotonic()
            content.eval()
            metrics, physical = observe(content, caches, state, basis, scale, projection, fusion, a_end, b_start, chunk)
            done = start + local
            diagnostics.update(physical=physical, optimize_seconds=optimize, evaluation_seconds=time.monotonic() - tic,
                               formal_write=False, matching_state_fixed=True)
            history.append(dict(global_step=done, round=number, phase='A', phase_step=local, metrics=metrics, diagnostics=diagnostics))
            print_scores('global%d/%d | R%d A %d/%d | lr %g | p,b FIXED' % (done,total,number,local,A_steps,A_lr), metrics)
            print('  A gradient %.6g | train response RMS %.6g | optimize %.3fs evaluate %.3fs' %
                  (diagnostics['A_gradient_norm'], physical['train']['response_rms'], optimize, diagnostics['evaluation_seconds']), flush=True)
            if local == A_steps:
                anchors[key]['after_A'] = metrics['A']['mse']
                anchors[key]['allowance'] = .1 * max(anchors[key]['round_start_A'] - metrics['A']['mse'], 0.)
                anchors[key]['limit'] = metrics['A']['mse'] + anchors[key]['allowance']
                store_endpoint(rd / 'A_end.pt', full_state(number, 'A', local, metrics, diagnostics))
                states['R%d_A_end' % number] = dict(metrics=metrics, diagnostics=diagnostics, global_step=done)
                write(rd / 'protection_anchor.json', anchors[key])
            if done % checkpoint_interval == 0 or local == A_steps or done == stop_after: checkpoint()
            if done == stop_after: return None
            content.train()
        content.eval().requires_grad_(False); optimizer_A.zero_grad(set_to_none=True)
        writer.train().requires_grad_(True)
        frozen_hash = module_hash(content); anchor_hash = tensor_hash(state)
        action_binding = dict(binding=binding, content_sha256=frozen_hash, applicability_sha256=anchor_hash, round=number)
        fixed = sealed_object(rd, 'frozen_actions', action_binding,
            lambda: freeze_actions(content, caches, state, basis, scale, projection, fusion, chunk, compact=True))
        feedback = sealed_object(rd, 'feedback', action_binding,
            lambda actions=fixed: collect_feedback(caches['train'], actions['train'], state, starts, a_end, b_start, chunk, window))
        torch.testing.assert_close(feedback['anchor'], state.cpu(), atol=0, rtol=0)
        if abs(feedback['A_mse'] - anchors[key]['after_A']) > 1e-10: raise ValueError('Feedback anchor MSE mismatch')
        write(rd / 'feedback_summary.json', feedback_summary(feedback))
        print('R%d Writer | FIXED post-A feedback | writable groups %d/%d | trial from same p,b; one final commit | no second order' %
              (number, feedback['write_groups'], groups), flush=True)
        for local in range(max(done - A_end, 0) + 1, W_steps + 1):
            tic = time.monotonic(); optimizer_W.zero_grad(set_to_none=True)
            diagnostics = writer_gradient(writer, feedback, state, caches['train'], fixed['train'],
                                          a_end, b_start, anchors[key]['limit'], chunk)
            checked_step(optimizer_W, writer); optimizer_W.zero_grad(set_to_none=True)
            optimize = time.monotonic() - tic; tic = time.monotonic()
            with torch.no_grad(): proposed = writer_candidate(writer, feedback, state).detach()
            metrics, physical = observe(content, caches, proposed, basis, scale, projection, fusion, a_end, b_start, chunk, fixed)
            done = A_end + local
            delta = proposed - state
            diagnostics.update(physical=physical, optimize_seconds=optimize, evaluation_seconds=time.monotonic() - tic,
                state_delta_rms=float(delta.square().mean().sqrt()), state_delta_norm_max=float(delta.norm(dim=-1).max()),
                applicability_norm_max=float(proposed[:, :-1].norm(dim=-1).max()), bias_abs_max=float(proposed[:, -1].abs().max()),
                write_groups=feedback['write_groups'], A_allowance=anchors[key]['allowance'], A_limit=anchors[key]['limit'],
                A_damage=metrics['A']['mse'] - anchors[key]['after_A'],
                A_excess=max(metrics['A']['mse'] - anchors[key]['limit'], 0.), formal_write=local == W_steps,
                score_role='candidate write; persistent p,b unchanged until phase end')
            history.append(dict(global_step=done, round=number, phase='Writer', phase_step=local, metrics=metrics, diagnostics=diagnostics))
            print_scores('global%d/%d | R%d Writer TRIAL %d/%d | lr %g' % (done,total,number,local,W_steps,W_lr), metrics)
            print('  Writer gradient %.6g | delta state RMS %.6g | A damage %.6g allowance %.6g excess %.6g | optimize %.3fs evaluate %.3fs | formal write %s' %
                  (diagnostics['Writer_gradient_norm'], diagnostics['state_delta_rms'], diagnostics['A_damage'],
                   diagnostics['A_allowance'], diagnostics['A_excess'], optimize, diagnostics['evaluation_seconds'], local == W_steps), flush=True)
            if local == W_steps:
                if module_hash(content) != frozen_hash or tensor_hash(state) != anchor_hash:
                    raise ValueError('A/content or write anchor changed during Writer training')
                mask = feedback['write_mask'].to(state.device)
                torch.testing.assert_close(proposed[~mask], state[~mask], atol=0, rtol=0)
                before = state; state = proposed.detach().clone(); commits += 1
                save(rd / 'write.pt', dict(before=before.cpu(), after=state.cpu(), delta=(state - before).cpu(),
                    write_mask=feedback['write_mask'], feedback_sha256=digest(rd / 'feedback.pt'),
                    formal_write_number=commits, Writer_actual_step=local, from_fixed_anchor=True))
                store_endpoint(rd / 'actual_end.pt', full_state(number, 'commit', local, metrics, diagnostics))
                states['R%d_end' % number] = dict(metrics=metrics, diagnostics=diagnostics, global_step=done)
                if best is None or metrics['validation']['mse'] < best['metrics']['validation']['mse'] - 1e-12:
                    best = full_state(number, 'commit', local, metrics, diagnostics)
                print_scores('R%d_FORMAL_WRITE_END' % number, metrics)
            if done % checkpoint_interval == 0 or local == W_steps or done == stop_after: checkpoint()
            if done == stop_after: return None
        del fixed, feedback
    if commits != rounds or best is None or best['round'] not in range(1,rounds+1) or best['phase'] != 'commit' or best['phase_step'] != W_steps:
        raise ValueError('Incomplete writes or selection outside complete round endpoints')
    states['RoundEnd_Validation_selected'] = dict(metrics=best['metrics'], diagnostics=best['diagnostics'], global_step=best['global_step'])
    save(folder / 'selected.pt', best); save(folder / 'states.pt', states)
    selection = dict(settings=settings, global_step=best['global_step'], round=best['round'], phase='commit',
        phase_step=best['phase_step'], metrics=best['metrics'], eligible_global_steps=[n*(A_steps+W_steps) for n in range(1,rounds+1)],
        actual_final_endpoint=total, formal_write_count=commits, new_A_updates=rounds * A_steps, new_Writer_updates=rounds * W_steps,
        historical_updates_reused=0, second_order=False,
        serving='conditional A + persistent content and applicability; no Writer/feedback/labels')
    write(folder / 'selection.json', selection); write(folder / 'history.json', history)
    return best, states, history, selection


def prepare_frozen_features(model,data,caches,args,scale,group,response_scale,out,device,binding,a_end,query_dim):
    from inference import query_features

    features_path = out / 'observables.pt'; seal_path = out / 'observables.json'
    if seal_path.exists():
        seal = read(seal_path)
        if seal['binding'] != binding or digest(features_path) != seal['sha256']: raise ValueError('Changed frozen observable cache')
        features = torch.load(features_path, map_location='cpu', weights_only=True)
        torch.testing.assert_close(features['group'], group.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(features['response_scale'], response_scale.cpu(), atol=0, rtol=0)
    elif features_path.exists():


        features = torch.load(features_path,map_location='cpu',weights_only=True)
        torch.testing.assert_close(features['group'],group.cpu(),atol=0,rtol=0)
        torch.testing.assert_close(features['response_scale'],response_scale.cpu(),atol=0,rtol=0)
        if set(features['splits'])!=set(caches) or set(features['queries'])!=set(caches):raise ValueError('Incomplete observable metadata')
        from cache import load_observables
        for sp,record in features['splits'].items():
            if record['file']!=sp+'_observables.npy':raise ValueError('Unexpected observable filename')
            value=load_observables(out,record)
            expected=(len(caches[sp]['target']),caches[sp]['ids'].shape[1],4*group.shape[1]+1+8*len(scale))
            if tuple(value.shape)!=expected or tuple(features['queries'][sp].shape)!=(expected[0],group.shape[1]):raise ValueError('Observable dimensions changed')
        print('RECOVER observables | all saved array hashes verified; no feature regeneration',flush=True)
    else:
        queries = query_features(model, data, device)
        features = dict(group=group.cpu(), response_scale=response_scale.cpu(), splits={}, queries=queries)
        for sp, cache in caches.items():
            from cache import open_observables, finish_observables
            matrix = None
            for lo in range(0, len(cache['target']), 128):
                rows = np.arange(lo, min(lo + 128, len(cache['target']))); ids = cache['ids'][rows].to(device)
                legal = data[sp].values(rows, ids, device).double()
                block = observable_features(queries[sp][rows].to(device), group[ids], cache['weights'][rows].to(device),
                    pooled(legal / scale), pooled(cache['prediction'][rows].to(device) / response_scale)).cpu()
                if matrix is None:matrix=open_observables(out,sp,(len(cache['target']),)+tuple(block.shape[1:]))
                matrix[lo:lo+len(rows)]=block.numpy()
                if lo % 2048 == 0: print('Fixed observable cache | %s | %d/%d' % (sp, lo + len(rows), len(cache['target'])), flush=True)
            features['splits'][sp] = finish_observables(out,sp,matrix)
        save(features_path, features)
    if not seal_path.exists():
        write(seal_path, dict(binding=binding, sha256=digest(features_path), A=[0, a_end],
            features='L2 hq,hg,product,absdiff,weight,legalValue4xC,originalM0prediction4xC',
            feature_dimension=features['splits']['train']['shape'][-1], response_scale=response_scale.cpu().tolist(),
            response_scale_fit='frozen MLP predictions on A only; distinct from original Value-bound scale'))
    from cache import load_observables
    features['splits']={sp:load_observables(out,record) for sp,record in features['splits'].items()}
    mapping=sealed_object(out,'query_mapping',dict(binding=binding,A=[0,a_end],dimension=query_dim),
        lambda:fit_query_map(features['queries']['train'],a_end,query_dim))
    return features,mapping


def execute_dynamic(model, data, labels, args, original, projection, fusion, scale, out, device, binding,
                    A_steps=100, W_steps=100, a_end=6337, b_start=6528, query_dim=32, synthetic_fixture=False, rounds=5, A_lr=.001, W_lr=.001, frozen_inputs=None, audit_only=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = projection.detach().double(); f = fusion.detach().double(); scale = scale.detach().double()
    if synthetic_fixture and not binding.get('synthetic_fixture'): raise ValueError('Explicit synthetic binding required')
    starts = np.asarray(data['train'].split.source_sample_index)
    horizon = p.shape[0]
    window = 96 + horizon
    if not synthetic_fixture:
        from horizons import role_split, expected_counts
        roles = role_split(expected_counts(horizon)['train'], horizon)
        from resources import allowed
        if not allowed(A_steps,W_steps,A_lr,W_lr,horizon) or (rounds,a_end,b_start,len(starts),query_dim) != (5,roles['A'][1],roles['B'][0],roles['B'][1],32):
            raise ValueError('Unsupported sweep case or chronological roles')
        if not np.all(np.diff(starts)==1) or starts[a_end-1]+window>starts[b_start]:
            raise ValueError('A/B windows overlap')
    caches = {sp: {k: v.detach().cpu().double() if v.is_floating_point() else v.detach().cpu()
              for k, v in c.items() if k in ('prediction', 'target', 'ids', 'weights')} for sp, c in original.items()}
    bank_keys = data['train'].keys[data['train'].bank.representatives].to(device)
    with torch.no_grad(): emb = model.encode_bank(bank_keys, args.encoding_chunk)
    group = normalized(emb.flatten(1)).to(scale)
    del bank_keys,emb
    if not synthetic_fixture and (tuple(group.shape) != (2048, model.architecture['retrieval_feature_dim']) or len(scale) != 7 or
            any(c['ids'].shape[1] != 20 for c in caches.values())): raise ValueError('Expected original2048/encoded_features/7/Top20')
    response_scale = data['train'].split.backbone_prediction[:a_end].double().reshape(-1, len(scale)).std(0, unbiased=False).clamp_min(1e-6).to(scale)
    if frozen_inputs is not None:
        features=frozen_inputs['features'];mapping=frozen_inputs['mapping']
        torch.testing.assert_close(features['group'],group.cpu(),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(features['response_scale'],response_scale.cpu(),atol=0,rtol=0)
        if mapping['fit_rows']!=[0,a_end] or mapping['dimension']!=query_dim:raise ValueError('Reused A-only PCA mapping differs')
        group=features['group'].to(scale)
        write(out/'frozen_input_reuse.json',dict(binding=binding,files=frozen_inputs['files'],
            source=frozen_inputs['source'],query_mapping_refit=False,query_observables_recomputed=False,
            evolved_content_or_Writer_loaded=False))
    else:
        if audit_only and not all((out/name).exists() for name in ('observables.pt','observables.json','query_mapping.pt','query_mapping.json')):
            raise ValueError('Audit-only requires existing frozen features/mapping')
        features,mapping=prepare_frozen_features(model,data,caches,args,scale,group,response_scale,out,device,binding,a_end,query_dim)
    for sp in caches: caches[sp]['observables']=features['splits'][sp]
    for sp in caches: caches[sp]['psi'] = query_coordinates(features['queries'][sp], mapping).cpu()
    if audit_only:raise ValueError('Use saved-state finish mode; extra replay is disabled')
    base=dict(format='feedback_state_writer_v1',query_mapping=mapping,scale=scale.cpu(),response_scale=response_scale.cpu(),
        projection=p.cpu(),fusion=f.cpu(),group_features=group.cpu(),horizon=horizon,binding=binding)
    save(out/'frozen_export_geometry.pt',base)
    selected,states,history,selection=solve(out/'trajectory',caches,scale,p,f,binding,starts,
        groups=len(group),A_steps=A_steps,W_steps=W_steps,a_end=a_end,b_start=b_start,
        synthetic=synthetic_fixture,chunk=64,rounds=rounds,A_lr=A_lr,W_lr=W_lr,window=window)
    from results import persist_results
    persist_results(out,selected,states,history,selection,base)
    print('SAVED round-end selected weight and original metrics | no post-training replay',flush=True)
    return selected,states
