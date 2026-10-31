"""Shared T5 graph/text encoder, claim planner, and plan-conditioned decoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5EncoderModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class ActionPredictor(nn.Module):
    """Three mutually exclusive action labels, conditional on a known DTI pair."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.d_model, 3)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(T5EncoderModel.from_pretrained(path, **kwargs))

    def forward(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        return self.head(masked_mean(hidden, tokens["attention_mask"]))

    def freeze(self):
        self.requires_grad_(False)
        self.eval()
        return self


class IncidenceLayer(nn.Module):
    """Bidirectional normalized source-to-assertion message passing."""
    def __init__(self, dim):
        super().__init__()
        self.to_graph = nn.Linear(dim, dim)
        self.to_source = nn.Linear(dim, dim)
        self.graph_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)

    def forward(self, sources, graph, incidence):
        if not sources.shape[0] or not graph.shape[0]:
            return sources, graph
        a = incidence.to(sources.dtype)
        graph_message = a @ sources / a.sum(1, keepdim=True).clamp_min(1)
        source_message = a.T @ graph / a.sum(0).unsqueeze(1).clamp_min(1)
        return (self.source_norm(sources + F.gelu(self.to_source(source_message))),
                self.graph_norm(graph + F.gelu(self.to_graph(graph_message))))


class EvidenceGenerator(nn.Module):
    def __init__(self, backbone, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        dim = backbone.config.d_model
        self.graph_layers = nn.ModuleList([IncidenceLayer(dim), IncidenceLayer(dim)])
        self.state_head = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 4))
        self.cite_query = nn.Linear(dim * 4, dim)
        self.cite_key = nn.Linear(dim, dim)
        self.state_embedding = nn.Linear(4, dim, bias=False)
        self.prediction_embedding = nn.Linear(3, dim, bias=False)
        self.plan_projection = nn.Linear(dim * 3, dim)
        self.freeze_backbone = freeze_backbone
        self.condition_on_plan = True
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, path, freeze_backbone=False, **kwargs):
        return cls(T5ForConditionalGeneration.from_pretrained(path, **kwargs), freeze_backbone)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode(self, tokens):
        hidden = self.backbone.get_encoder()(**tokens).last_hidden_state
        return hidden, masked_mean(hidden, tokens["attention_mask"])

    @staticmethod
    def attend(query, values):
        if not values.shape[0]:
            return torch.zeros_like(query)
        weights = (query @ values.T / math.sqrt(query.shape[-1])).softmax(-1)
        return weights @ values

    def plan(self, features, fixed_prediction):
        pair_hidden, pair = self.encode(features["pair"])
        _, claims = self.encode(features["claim"])
        dim = pair.shape[-1]
        if features["source"] is None:
            source_hidden = pair.new_empty((0, 0, dim))
            sources = pair.new_empty((0, dim))
        else:
            source_hidden, sources = self.encode(features["source"])
        if features["graph"] is None:
            graph = pair.new_empty((0, dim))
        else:
            _, graph = self.encode(features["graph"])
        for layer in self.graph_layers:
            sources, graph = layer(sources, graph, features["incidence"])
        joint = torch.cat([claims, pair.expand_as(claims), self.attend(claims, sources), self.attend(claims, graph)], -1)
        state_logits = self.state_head(joint)
        cite_logits = self.cite_query(joint) @ self.cite_key(sources).T / math.sqrt(dim)
        weights = cite_logits.sigmoid()
        citation_pool = weights @ sources / weights.sum(-1, keepdim=True).clamp_min(1)
        # The decoder sees predicted soft states and citations, never oracle states.
        plans = self.plan_projection(torch.cat([claims, self.state_embedding(state_logits.softmax(-1)), citation_pool], -1))
        if not self.condition_on_plan:
            plans = plans[:0]
        memory = [pair_hidden, graph.unsqueeze(0), sources.unsqueeze(0), plans.unsqueeze(0),
                  self.prediction_embedding(fixed_prediction.detach()).reshape(1, 1, dim)]
        masks = [features["pair"]["attention_mask"],
                 torch.ones((1, len(graph) + len(sources) + len(plans) + 1), dtype=torch.long, device=pair.device)]
        if features["source"] is not None:
            memory.append(source_hidden.reshape(1, -1, dim))
            masks.append(features["source"]["attention_mask"].reshape(1, -1))
        return {"state_logits": state_logits, "citation_logits": cite_logits,
                "memory": torch.cat(memory, 1), "memory_mask": torch.cat(masks, 1)}

    def forward(self, features, fixed_prediction, text_labels=None):
        output = self.plan(features, fixed_prediction)
        if text_labels is not None:
            decoded = self.backbone(encoder_outputs=BaseModelOutput(last_hidden_state=output["memory"]),
                                    attention_mask=output["memory_mask"], labels=text_labels, use_cache=False)
            output["text_loss"] = decoded.loss
        return output

    @torch.no_grad()
    def generate(self, features, fixed_prediction, **kwargs):
        output = self.plan(features, fixed_prediction)
        ids = self.backbone.generate(encoder_outputs=BaseModelOutput(last_hidden_state=output["memory"]),
                                     attention_mask=output["memory_mask"], **kwargs)
        return ids, output


def paired_loss(before, after, labels_before, labels_after, citations_before, citations_after, weights=None):
    weights = weights or {}
    b, a = before["state_logits"], after["state_logits"]
    changed = labels_before != labels_after
    zero = b.sum() * 0
    losses = {"state": (F.cross_entropy(b, labels_before) + F.cross_entropy(a, labels_after)) / 2}
    pb, pa = b.softmax(-1), a.softmax(-1)
    target_delta = F.one_hot(labels_after, 4).float() - F.one_hot(labels_before, 4).float()
    losses["change"] = F.smooth_l1_loss((pa - pb)[changed], target_delta[changed]) if changed.any() else zero
    unchanged = ~changed
    if unchanged.any():
        x, y = pb[unchanged].clamp_min(1e-8), pa[unchanged].clamp_min(1e-8)
        losses["keep"] = ((x - y) * (x.log() - y.log())).sum(-1).mean() / 2
    else:
        losses["keep"] = zero
    citation_terms = [F.binary_cross_entropy_with_logits(out["citation_logits"], gold)
                      for out, gold in [(before, citations_before), (after, citations_after)] if gold.numel()]
    losses["cite"] = torch.stack(citation_terms).mean() if citation_terms else zero
    losses["text"] = (before["text_loss"] + after["text_loss"]) / 2
    total = sum(weights.get(key, 1.0) * value for key, value in losses.items())
    return total, losses
