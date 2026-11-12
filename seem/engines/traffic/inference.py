from __future__ import annotations
import math
import torch
from evolution import normalized, pooled, observable_features, temporal_basis, decode
from evolution import load_content, query_coordinates
import numpy as np
from runtime import adaptive


class StateMemory:
    def __init__(self, artifact, device='cpu'):
        if artifact.get('format') != 'feedback_state_writer_v1': raise ValueError('Wrong state-memory artifact')
        self.content = load_content(artifact['content'], device).eval().requires_grad_(False)
        self.state = artifact['applicability'].to(device).double()
        self.mapping = dict(artifact['query_mapping'])
        for name in ('center', 'axes', 'deviation'): self.mapping[name] = self.mapping[name].to(device)
        self.scale = artifact['scale'].to(device).double()
        self.response_scale = artifact['response_scale'].to(device).double()
        self.group_features = artifact['group_features'].to(device).double()
        self.projection = artifact['projection'].to(device).double()
        self.fusion = artifact['fusion'].to(device).double()
        self.horizon = artifact['horizon']; self.basis = temporal_basis(self.horizon, device)
        self._checked = False

    @torch.no_grad()
    def predict(self, model, query_keys, histories, backbone_predictions, bank_embeddings,
                group_labels, low, high, legal_value_reader):
        if not self._checked:
            torch.testing.assert_close(model.fusion.retrieval_pred[0].weight.double(), self.projection, atol=0, rtol=0)
            torch.testing.assert_close(model.fusion.linear_pred.weight[:, self.horizon:].double(), self.fusion, atol=0, rtol=0)
            torch.testing.assert_close(normalized(bank_embeddings.flatten(1)), self.group_features, atol=1e-6, rtol=1e-6)
            self._checked = True
        ret = model.retrieve(query_keys, bank_embeddings, group_labels, len(self.group_features), low, high)
        ids = ret['group_ids']; weights = ret['weights'].double()
        legal = legal_value_reader(ids)
        m0 = model(histories, backbone_predictions, legal, ret['weights'])
        query = normalized(ret['query_embeddings'].flatten(1))
        features = observable_features(query, self.group_features[ids], weights,
            pooled(legal.double() / self.scale), pooled(m0.double() / self.response_scale))
        coefficients = self.content(features, ids)
        psi = query_coordinates(query, self.mapping, self.scale.device)
        selected = self.state[ids]
        alpha = (selected[..., -1] + (psi[:, None] * selected[..., :-1]).sum(-1) / math.sqrt(psi.shape[-1])).sigmoid()
        delta = decode(coefficients, self.basis, self.scale) * alpha[..., None, None]
        pred = model(histories, backbone_predictions, legal + delta.to(legal.dtype), ret['weights'])
        return pred, ret, coefficients, delta, alpha


@torch.no_grad()
def query_features(model, data, device, chunk=128):
    result = {}
    for sp, ds in data.items():
        values = []
        for lo in range(0, ds.split.count, chunk):
            rows = np.arange(lo, min(lo + chunk, ds.split.count))
            query, _, _ = ds.queries(rows, device)
            values.append(normalized(model.encoder(query).flatten(1)).cpu())
        result[sp] = torch.cat(values)
    return result


def load_selected(path, device='cpu'):
    return StateMemory(torch.load(path, map_location='cpu', weights_only=True), device)


@torch.no_grad()
def predict(model,data,labels,memory,split,rows,bank_embeddings=None):

    device=next(model.parameters()).device;rows=np.asarray(rows,dtype=np.int64);ds=data[split]
    if bank_embeddings is None:
        keys=data['train'].keys[data['train'].bank.representatives].to(device)
        bank_embeddings=model.encode_bank(keys,128)
    query,low,high=ds.queries(rows,device);history,backbone=ds.forecast_inputs(rows,device)
    return memory.predict(model,query,history,backbone,bank_embeddings,labels,low,high,
        lambda ids:ds.values(rows,ids,device))


