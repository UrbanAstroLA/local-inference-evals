# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 MLA layer ownership (recipe patch; see port-glm53-layer-owner.py)."""

from __future__ import annotations

import functools
import json
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_GLM53_MLA_OWNERS"


@functools.cache
def _parse(spec: str, mla_layers: tuple[int, ...], world: int) -> dict[int, int]:
    spec = spec.strip()
    if spec in {"", "tp", "off", "none"}:
        return {}
    if spec.startswith("split:"):
        boundary = int(spec.split(":", 1)[1])
        if world != 2:
            raise ValueError(f"{ENV}=split:N requires TP=2, got TP={world}")
        owners = {layer: int(layer >= boundary) for layer in mla_layers}
    elif spec.startswith("map:"):
        owners = {}
        for item in spec.split(":", 1)[1].split(","):
            layer, rank = item.split("=")
            owners[int(layer)] = int(rank)
        missing = sorted(set(mla_layers) - set(owners))
        extra = sorted(set(owners) - set(mla_layers))
        if missing or extra:
            raise ValueError(
                f"{ENV} map must name every MLA layer exactly: "
                f"missing={missing} extra={extra}"
            )
    else:
        raise ValueError(f"{ENV} must be tp, split:N or map:L=R,...; got {spec!r}")
    bad = {layer: rank for layer, rank in owners.items() if not 0 <= rank < world}
    if bad:
        raise ValueError(f"{ENV} owner ranks out of range for TP={world}: {bad}")
    counts = [sum(rank == r for rank in owners.values()) for r in range(world)]
    if min(counts) == 0:
        raise ValueError(
            f"{ENV} leaves a rank without MLA layers {counts}; GLM's KDA state "
            "shares MLA slot tensors, so every rank needs at least one"
        )
    return owners


def _mla_layers(hf_text_config) -> tuple[int, ...]:
    return tuple(
        i
        for i in range(hf_text_config.num_hidden_layers)
        if not hf_text_config.is_kda_layer(i)
    )


def mla_owner_map(vllm_config) -> dict[int, int]:
    """Return {mla_layer: owner_rank}; empty when ownership is disabled."""
    spec = os.environ.get(ENV, "tp")
    hf_text_config = vllm_config.model_config.hf_text_config
    if getattr(hf_text_config, "model_type", None) not in {
        "glm5_next",
        "glm5_next_text",
    }:
        return {}
    parallel = vllm_config.parallel_config
    owners = _parse(
        spec, _mla_layers(hf_text_config), int(parallel.tensor_parallel_size)
    )
    if owners and int(parallel.decode_context_parallel_size) != 1:
        raise ValueError(f"{ENV}={spec} replaces DCP; set DCP=1")
    if owners and int(parallel.pipeline_parallel_size) != 1:
        raise ValueError(f"{ENV}={spec} requires PP=1")
    return owners


def mla_owner(layer_idx: int, vllm_config) -> int | None:
    return mla_owner_map(vllm_config).get(layer_idx)


def mla_counts_per_rank(vllm_config) -> list[int] | None:
    owners = mla_owner_map(vllm_config)
    if not owners:
        return None
    world = int(vllm_config.parallel_config.tensor_parallel_size)
    return [sum(rank == r for rank in owners.values()) for r in range(world)]


class Glm5NextRemoteAttention(torch.nn.Module):
    """Placeholder for an MLA layer owned by the other TP rank.

    Holds no weights and registers no KV cache. The weight loader treats its
    prefix as missing (see ``is_pp_missing_parameter``) so the owner-only
    checkpoint tensors are skipped on this rank.
    """

    glm53_owner_mode = "remote"

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        return torch.zeros_like(hidden_states)


def attention_forward(attn, hidden_states: torch.Tensor, positions: torch.Tensor):
    """Run attention; complete owned layers with the TP all-reduce."""
    mode = getattr(attn, "glm53_owner_mode", None)
    if mode is None:
        return attn(hidden_states=hidden_states, positions=positions)
    from vllm.distributed import tensor_model_parallel_all_reduce

    out = attn(hidden_states=hidden_states, positions=positions)
    return tensor_model_parallel_all_reduce(out)


def build_mla_attention(cls, **kwargs):
    """Construct an owned (unsharded) MLA layer or its remote placeholder."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.models.utils import extract_layer_index

    vllm_config = kwargs["vllm_config"]
    layer_idx = extract_layer_index(kwargs["prefix"])
    owner = None
    if layer_idx < vllm_config.model_config.hf_text_config.num_hidden_layers:
        owner = mla_owner(layer_idx, vllm_config)
    if owner is None:
        return cls(**kwargs)
    if get_tensor_model_parallel_rank() != owner:
        return Glm5NextRemoteAttention()
    module = cls(owner_local=True, **kwargs)
    module.glm53_owner_mode = "local"
    return module


EMBED_ENV = "VLLM_GLM53_EMBED_SPLIT"


def embed_split(vllm_config) -> list[float] | None:
    """Per-rank fractions of the vocabulary rows, or None for even TP."""
    spec = os.environ.get(EMBED_ENV, "").strip()
    if spec in {"", "tp", "off", "none"}:
        return None
    world = int(vllm_config.parallel_config.tensor_parallel_size)
    parts = [float(x) for x in spec.split(",")]
    if len(parts) == world - 1:
        parts.append(1.0 - sum(parts))
    if len(parts) != world or any(x < 0 for x in parts) or abs(sum(parts) - 1) > 1e-6:
        raise ValueError(
            f"{EMBED_ENV}={spec} must give {world - 1} or {world} non-negative "
            "rank fractions summing to 1"
        )
    return parts


class Glm5NextSplitEmbedding(torch.nn.Module):
    """Token embedding whose vocabulary rows are split unevenly across TP.

    Rank r holds rows [start_r, end_r). Out-of-range ids contribute zeros and
    the TP all-reduce combines the ranks, exactly like vLLM's even
    vocab-parallel lookup. Uneven fractions let the recipe balance the two
    GPUs' free memory for the KV pool at essentially no lookup cost.
    """

    def __init__(self, num_embeddings, embedding_dim, fractions, rank, dtype):
        super().__init__()
        bounds = [0]
        for fraction in fractions[:-1]:
            bounds.append(min(num_embeddings, round(bounds[-1] + fraction * num_embeddings)))
        bounds.append(num_embeddings)
        self.start, self.end = bounds[rank], bounds[rank + 1]
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        rows = max(self.end - self.start, 0)
        self.glm53_owner_mode = "local" if rows else "remote"
        self.weight = torch.nn.Parameter(
            torch.empty(rows, embedding_dim, dtype=dtype), requires_grad=False
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        assert loaded_weight.shape[0] >= self.end, loaded_weight.shape
        param.data.copy_(loaded_weight[self.start : self.end])

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce

        ids = input_.long()
        in_range = (ids >= self.start) & (ids < self.end)
        if self.end > self.start:
            local = torch.nn.functional.embedding(
                torch.where(in_range, ids - self.start, torch.zeros_like(ids)),
                self.weight,
            )
            local = local.masked_fill(~in_range.unsqueeze(-1), 0)
        else:
            local = torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self.weight.dtype,
                device=ids.device,
            )
        return tensor_model_parallel_all_reduce(local)


def build_embedding(cls, num_embeddings: int, embedding_dim: int, *, prefix: str, vllm_config):
    """Build vLLM's even vocab-parallel embedding or the uneven split one."""
    from vllm.distributed import get_tensor_model_parallel_rank

    fractions = embed_split(vllm_config)
    if fractions is None:
        return cls(num_embeddings, embedding_dim, prefix=prefix)
    module = Glm5NextSplitEmbedding(
        num_embeddings,
        embedding_dim,
        fractions,
        get_tensor_model_parallel_rank(),
        vllm_config.model_config.dtype,
    )
    if module.glm53_owner_mode == "remote":
        # No rows: keep the loader from resolving the checkpoint tensor here.
        del module.weight
    return module


_COMPONENTS = (
    ("vision", ("visual.",)),
    ("embed", ("embed_tokens",)),
    ("lm_head", ("lm_head",)),
    ("routed_experts", (".mlp.experts.",)),
    ("shared_experts", ("shared_expert",)),
    ("router", (".mlp.gate.",)),
)


def _classify(name: str, mla: set[int], kda: set[int]) -> str:
    for component, needles in _COMPONENTS:
        if any(needle in name for needle in needles):
            return component
    marker = ".layers."
    if marker in name:
        layer = int(name.split(marker, 1)[1].split(".", 1)[0])
        if ".self_attn." in name:
            return "mla_attention" if layer in mla else "kda_attention"
        if ".mlp." in name:
            return "dense_mlp"
        return "layer_other"
    return "other"


def log_placement_ledger(model, vllm_config, extra_models=()) -> None:
    """Log a per-rank byte ledger of resident parameters and buffers."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        hf_text_config = vllm_config.model_config.hf_text_config
        mla = set(_mla_layers(hf_text_config))
        kda = set(range(hf_text_config.num_hidden_layers)) - mla
        ledger: dict[str, int] = {}
        seen: set[int] = set()
        sources = [("", model)] + [(f"{tag}.", m) for tag, m in extra_models]
        for tag, module in sources:
            if module is None:
                continue
            tensors = list(module.named_parameters()) + list(module.named_buffers())
            for name, tensor in tensors:
                if tensor.device.type != "cuda":
                    continue
                storage = tensor.untyped_storage()
                key = storage.data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                component = "draft" if tag else _classify(name, mla, kda)
                ledger[component] = ledger.get(component, 0) + storage.nbytes()
        owners = mla_owner_map(vllm_config)
        rank = get_tensor_model_parallel_rank()
        record = {
            "rank": rank,
            "mla_owned": sorted(k for k, v in owners.items() if v == rank),
            "gib": {k: round(v / 2**30, 3) for k, v in sorted(ledger.items())},
            "total_gib": round(sum(ledger.values()) / 2**30, 3),
            "torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        }
        logger.info("glm53-placement ledger %s", json.dumps(record, sort_keys=True))
    except Exception as exc:  # diagnostics must never break serving
        logger.warning("glm53-placement ledger unavailable: %s", exc)
