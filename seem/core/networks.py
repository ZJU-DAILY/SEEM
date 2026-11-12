from __future__ import annotations
import utils
import copy
import math
import time
from typing import Sequence, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from utils import cpu_copy


class ResidualMLP(torch.nn.Module):
    def __init__(self,seq_len=96,pred_len=96,hidden=512,dropout=0.1,eps=1e-5,**unused):
        super().__init__()
        self.seq_len=seq_len;self.pred_len=pred_len;self.eps=eps
        self.input=torch.nn.Linear(seq_len,hidden)
        self.block=torch.nn.Sequential(torch.nn.Linear(hidden,hidden),torch.nn.GELU(),torch.nn.Linear(hidden,hidden),torch.nn.GELU())
        self.output=torch.nn.Sequential(torch.nn.Dropout(dropout),torch.nn.Linear(hidden,pred_len))
    def forward(self,x,*unused):
        mean=x.mean(1,keepdim=True).detach()
        std=(x.var(1,keepdim=True,unbiased=False)+self.eps).sqrt().detach()
        h=self.input(((x-mean)/std).transpose(1,2))
        return self.output(h+self.block(h)).transpose(1,2)*std+mean

class ForecastFusion(nn.Module):


    def __init__(self, pred_len: int, periods: Sequence[int], use_memory: bool):
        super().__init__()
        self.pred_len = int(pred_len)
        self.periods = tuple(int(period) for period in periods)
        self.use_memory = bool(use_memory)
        self.retrieval_pred = nn.ModuleList(
            [nn.Linear(self.pred_len // period, self.pred_len) for period in self.periods]
        )
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)


        with torch.no_grad():
            self.linear_pred.weight.zero_()
            self.linear_pred.bias.zero_()
            identity = torch.arange(self.pred_len)
            self.linear_pred.weight[identity, identity] = 1.0

    def forward(
        self,
        histories: torch.Tensor,
        backbone: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if histories.ndim != 3 or backbone.ndim != 3:
            raise ValueError("histories and backbone must have shape [batch, steps, channels].")
        if backbone.shape[0] != histories.shape[0] or backbone.shape[1] != self.pred_len:
            raise ValueError("backbone prediction does not match batch/prediction length.")
        offset = histories[:, -1:, :].detach()
        backbone_delta = backbone - offset

        if self.use_memory:
            if prototypes is None or prototypes.ndim != 4:
                raise ValueError("Memory fusion requires prototypes [scales, batch, pred_len, channels].")
            if prototypes.shape[0] != len(self.periods) or prototypes.shape[1] != histories.shape[0]:
                raise ValueError("Prototype scales/batch do not match the fusion input.")
            if prototypes.shape[2] != self.pred_len or prototypes.shape[3] != histories.shape[2]:
                raise ValueError("Prototype prediction/channel shape does not match the fusion input.")
            projected_scales = []
            for scale_index, prototype in enumerate(prototypes):
                period = self.periods[scale_index]
                batch_size, _, channels = prototype.shape
                compressed = prototype.reshape(batch_size, self.pred_len // period, period, channels)
                compressed = compressed[:, :, 0, :]
                projected = self.retrieval_pred[scale_index](compressed.permute(0, 2, 1))
                projected_scales.append(projected.permute(0, 2, 1))
            retrieval_delta = torch.stack(projected_scales, dim=1).sum(dim=1)
        else:


            retrieval_delta = torch.zeros_like(backbone_delta)

        combined = torch.cat([backbone_delta, retrieval_delta], dim=1)
        forecast_delta = self.linear_pred(combined.permute(0, 2, 1)).permute(0, 2, 1)
        return forecast_delta + offset

TAU_R=.1

def group_max(scores, labels, groups, low, high):

    if scores.shape[1] != len(labels):
        raise ValueError('Full bank member axis mismatch.')
    positions = torch.arange(len(labels), device=scores.device)[None]
    forbidden = (positions >= low[:, None]) & (positions < high[:, None])
    masked = scores.masked_fill(forbidden, -torch.inf)
    result = scores.new_full((len(scores), groups), -torch.inf)
    result = result.scatter_reduce(1, labels[None].expand_as(scores), masked, reduce='amax', include_self=True)
    if torch.isnan(result).any() or (torch.isfinite(result).sum(1) < 20).any():
        raise ValueError('Invalid group scores or fewer than 20 legal groups.')
    return result

def select(group_scores):

    ids = torch.argsort(group_scores, dim=1, descending=True, stable=True)[:, :20]
    selected = group_scores.gather(1, ids)
    if not torch.isfinite(selected).all():
        raise ValueError('Selected an invalid group.')
    weights = (selected/TAU_R).softmax(1)
    return ids, weights

class GlobalReader(nn.Module):
    def __init__(self, fusion, encoder=None):
        super().__init__()
        self.fusion = copy.deepcopy(fusion)
        self.encoder = copy.deepcopy(encoder)

    def encode_bank(self, bank_keys, chunk=2048):
        if self.encoder is None:
            return bank_keys

        return torch.cat([self.encoder(bank_keys[start:start+chunk]) for start in range(0, len(bank_keys), chunk)], 0)

    def retrieve(self, query_keys, bank_embeddings, labels, groups, low, high):
        query = query_keys if self.encoder is None else self.encoder(query_keys)
        scores = query @ bank_embeddings.T
        grouped = group_max(scores, labels, groups, low, high)
        ids, weights = select(grouped)
        return dict(group_scores=grouped, group_ids=ids, weights=weights,
                    member_columns=scores.shape[1], query_embeddings=query)

    def forward(self, history, backbone, values, weights):
        prototype = (values*weights[..., None, None]).sum(1)[None]
        prediction = self.fusion(history, backbone, prototype)
        if not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite forecast.')
        return prediction

DEFAULT=dict(lambda_pcl=.1,tau_p=.05,top_k=20,tau_r=.1)

def hyperparameters(values):
    if set(values)!=set(DEFAULT):raise ValueError('Exactly four hyperparameters are required.')
    out={k:int(values[k]) if k=='top_k' else float(values[k]) for k in DEFAULT}
    if float(values['top_k'])!=out['top_k'] or out['top_k'] not in (5,10,20,40):raise ValueError('Invalid top_k.')
    if not all(math.isfinite(v) and v>0 for v in out.values()):raise ValueError('Hyperparameters must be finite positive.')
    return out

class TunableReader(GlobalReader):
    def __init__(self,fusion,encoder,hparams):
        super().__init__(fusion,encoder);self.hparams=hyperparameters(hparams)

    def retrieve(self,query_keys,bank_embeddings,labels,groups,low,high):
        query=self.encoder(query_keys);scores=query@bank_embeddings.T
        grouped=group_max(scores,labels,groups,low,high)
        k=self.hparams['top_k']
        ids=torch.argsort(grouped,dim=1,descending=True,stable=True)[:,:k]
        selected=grouped.gather(1,ids)
        if ids.shape[1]!=k or not torch.isfinite(selected).all():raise ValueError('Fewer than K legal distinct groups.')
        weights=(selected/self.hparams['tau_r']).softmax(1)
        return dict(group_scores=grouped,group_ids=ids,weights=weights,member_columns=scores.shape[1],query_embeddings=query)

    def pcl(self,scores,positive):
        if not torch.isfinite(scores.gather(1,positive[:,None])).all():raise ValueError('Illegal Hard PCL positive.')
        return F.cross_entropy(scores/self.hparams['tau_p'],positive)

class SemanticTokens(nn.Module):
    def __init__(self, history_size, response_size, channels, token_dim):
        super().__init__()
        if history_size % channels or response_size % (3 * channels):
            raise ValueError('Invalid flattened History/response dimensions.')
        self.history_size, self.response_size, self.channels = history_size, response_size, channels
        lengths = [history_size // channels] + [response_size // (3 * channels)] * 3

        self.views = nn.ModuleList([nn.Sequential(nn.Linear(length, 32), nn.GELU(),
                                                  nn.Linear(32, token_dim), nn.LayerNorm(token_dim))
                                    for length in lengths])

    def forward(self, keys):
        if keys.ndim != 2 or keys.shape[1] != self.history_size + self.response_size:
            raise ValueError('Wrong packed History/response Key shape.')
        n, c = len(keys), self.channels
        history = keys[:, :self.history_size].reshape(n, -1, c).transpose(1, 2)
        response = keys[:, self.history_size:].reshape(n, 3, -1, c).permute(0, 1, 3, 2)
        return torch.stack([self.views[0](history)] + [self.views[v + 1](response[:, v]) for v in range(3)], dim=1)

class AxialMatcher(nn.Module):
    def __init__(self, channels, token_dim=8, width=16):
        super().__init__()

        self.interaction = nn.Parameter(torch.empty(4, 4, token_dim, width))
        for view in range(4):
            for part in range(4):
                nn.init.xavier_uniform_(self.interaction[view, part])
        self.interaction_bias = nn.Parameter(torch.zeros(4, width))
        self.channel_identity = nn.Parameter(torch.randn(4, channels, width) * .01)
        self.channel_mix = nn.Parameter(torch.eye(channels)[None].repeat(4, 1, 1) + torch.randn(4, channels, channels) * .01)
        self.channel_feature = nn.Linear(width, width)
        self.view_mix = nn.Parameter(torch.eye(4) + torch.randn(4, 4) * .01)
        self.view_feature = nn.Linear(width, width)
        self.channel_norm = nn.LayerNorm(width)
        self.view_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.normal_(self.head[-1].weight, std=.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query, members):

        q = query[:, None]; m = members[None]
        weights = self.interaction
        project = lambda x, part: torch.einsum('...vcd,vdh->...vch', x, weights[:, part])
        h = project(q, 0) + project(m, 1) + project((q - m).abs(), 2) + project(q * m, 3)
        h = F.gelu(h + self.interaction_bias[:, None, :] + self.channel_identity)
        channel = torch.einsum('vck,...vkh->...vch', self.channel_mix, self.channel_feature(h))
        h = self.channel_norm(h + F.gelu(channel))
        view = torch.einsum('vu,...uch->...vch', self.view_mix, self.view_feature(h))
        h = self.view_norm(h + F.gelu(view))
        return torch.tanh(self.head(h.mean(dim=(-3, -2))).squeeze(-1))

class StructuredBehavioralReader(TunableReader):
    def __init__(self, original, response_size, channels, token_dim=8, width=16,
                 member_chunk=128, checkpoint_matching=True):
        super().__init__(original.fusion, original.encoder, original.hparams)
        history_size = original.encoder.layers[0].in_features
        self.encoder = SemanticTokens(history_size, response_size, channels, token_dim)
        self.matcher = AxialMatcher(channels, token_dim, width)
        self.member_chunk = member_chunk
        self.checkpoint_matching = checkpoint_matching
        self.architecture = dict(history_size=history_size, response_size=response_size, channels=channels,
                                 token_dim=token_dim, width=width)

    def score_members(self, query, bank):
        blocks = []
        for start in range(0, len(bank), self.member_chunk):
            members = bank[start:start + self.member_chunk]
            if torch.is_grad_enabled() and self.checkpoint_matching:

                scores = checkpoint(self.matcher, query, members, use_reentrant=False)
            else:
                scores = self.matcher(query, members)
            blocks.append(scores)
        return torch.cat(blocks, dim=1)

    def retrieve(self, query_keys, bank_embeddings, labels, groups, low, high):
        query = self.encoder(query_keys)
        scores = self.score_members(query, bank_embeddings)
        if hasattr(self, 'representative_rows'):
            if scores.shape[1] != groups:
                raise ValueError('Single-Key bank must score exactly one address per group')
            reps = self.representative_rows
            legal = (reps[None] < low[:, None]) | (reps[None] >= high[:, None])
            grouped = scores.masked_fill(~legal, -torch.inf)
        else:
            grouped = group_max(scores, labels, groups, low, high)
        k = self.hparams['top_k']
        ids = torch.argsort(grouped, dim=1, descending=True, stable=True)[:, :k]
        selected = grouped.gather(1, ids)
        if ids.shape[1] != k or not torch.isfinite(selected).all():
            raise ValueError('Insufficient legal groups.')
        return dict(group_scores=grouped, group_ids=ids, weights=(selected / self.hparams['tau_r']).softmax(1),
                    member_columns=len(bank_embeddings), query_embeddings=query)

class BilinearMatcher(nn.Module):
    def __init__(self, channels, token_dim=8, rank=64, score_scale=1.0):
        super().__init__()
        self.channels = channels; self.token_dim = token_dim; self.rank = rank
        self.score_scale = score_scale
        self.collect_score_diagnostics = False
        self.score_records = []
        size = 4 * channels * token_dim
        self.query_projection = nn.Linear(size, rank, bias=False)
        self.memory_projection = nn.Linear(size, rank, bias=False)
        nn.init.normal_(self.query_projection.weight, std=0.02)
        nn.init.normal_(self.memory_projection.weight, std=0.02)

    def forward(self, query, members):
        if (query.ndim != 4 or members.ndim != 4 or query.shape[1:] != members.shape[1:] or
                tuple(query.shape[1:]) != (4, self.channels, self.token_dim)):
            raise ValueError('Expected original SemanticTokens [N,4,C,8] on both sides')
        q = self.query_projection(query.flatten(1))
        m = self.memory_projection(members.flatten(1))
        raw = (q @ m.T) / math.sqrt(self.rank)

        pre_tanh = raw if self.score_scale == 1.0 else raw * self.score_scale
        score = torch.tanh(pre_tanh)
        if self.collect_score_diagnostics:
            from framework import score_record
            self.score_records.append(score_record(raw, pre_tanh, score))
        return score

class BilinearReader(StructuredBehavioralReader):
    def score_members(self, query, bank):

        profile = getattr(self, '_profile_matcher', False)
        if profile and query.device.type == 'cuda': torch.cuda.synchronize(query.device)
        began = time.perf_counter() if profile else None
        scores = self.matcher(query, bank)
        if profile:
            if query.device.type == 'cuda': torch.cuda.synchronize(query.device)
            self.last_matcher_seconds = time.perf_counter() - began
        return scores

class Content(nn.Module):
    def __init__(self, center, deviation, groups, channels, content_dim=32, hidden=64):
        super().__init__()
        self.groups = groups; self.channels = channels; self.content_dim = content_dim; self.hidden = hidden
        self.register_buffer('center', center.detach().float().clone())
        self.register_buffer('deviation', deviation.detach().float().clone())
        self.codes = nn.Parameter(torch.zeros(groups, content_dim, device=center.device))
        self.net = nn.Sequential(nn.Linear(len(center) + content_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 21 * channels)).to(center.device)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, ids):
        x = (features.float() - self.center) / self.deviation
        return self.net(torch.cat((x, self.codes[ids]), -1)).reshape(*features.shape[:-1], 21, self.channels)

    def artifact(self):
        return dict(state=cpu_copy(self.state_dict()), groups=self.groups, channels=self.channels,
                    content_dim=self.content_dim, hidden=self.hidden, feature_dim=len(self.center))

def load_content(artifact, device='cpu'):
    state = artifact['state']
    content = Content(state['center'].to(device), state['deviation'].to(device), artifact['groups'],
                      artifact['channels'], artifact['content_dim'], artifact['hidden'])
    content.load_state_dict(state); return content

class Writer(nn.Module):
    def __init__(self, input_dim=243, state_dim=33, hidden=64, device='cpu'):
        super().__init__(); self.input_dim = input_dim; self.state_dim = state_dim; self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, state_dim)).to(device)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feedback): return self.net(feedback.float())

    def artifact(self):
        return dict(state=cpu_copy(self.state_dict()), input_dim=self.input_dim,
                    state_dim=self.state_dim, hidden=self.hidden)
