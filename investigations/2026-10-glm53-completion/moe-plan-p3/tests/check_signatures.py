#!/usr/bin/env python3
"""CPU-only: every vLLM / B12x callable the harness uses exists in this
image with the expected signature, and the served-path source lines the
harness mirrors are present verbatim.

Usage: check_signatures.py ENGINE OUT.json      (inside the image, CPU only)
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import sys

ENGINE = sys.argv[1]
OUT = sys.argv[2]
res: dict = dict(engine=ENGINE, checks=[], failures=[], def_lines={})


def check(name, ok, detail=""):
    res["checks"].append(dict(name=name, ok=bool(ok), detail=str(detail)[:400]))
    if not ok:
        res["failures"].append(name)


def params(fn):
    return [p.name for p in inspect.signature(fn).parameters.values()]


def expect_params(name, fn, expected):
    try:
        got = params(fn)
    except Exception as exc:  # noqa: BLE001
        check(name, False, repr(exc))
        return
    check(name, got == expected, f"got={got} expected={expected}")


import torch  # noqa: E402
import vllm.model_executor.layers.quantization.exl3 as exl3  # noqa: E402
import vllm.model_executor.parameter as vparam  # noqa: E402
from vllm.model_executor.layers.fused_moe import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map  # noqa: E402

M = exl3.Exl3MoEMethod
expect_params("Exl3Config.from_config", exl3.Exl3Config.from_config, ["config"])
expect_params("Exl3Config.maybe_update_config", exl3.Exl3Config.maybe_update_config,
              ["self", "model_name", "hf_config", "revision"])
expect_params("Exl3MoEMethod.__init__", M.__init__, ["self", "quant_config", "moe"])
expect_params("Exl3MoEMethod.create_weights", M.create_weights,
              ["self", "layer", "num_experts", "hidden_size", "intermediate_size_per_partition",
               "params_dtype", "extra_weight_attrs"])
expect_params("Exl3MoEMethod.process_weights_after_loading", M.process_weights_after_loading,
              ["self", "layer"])
expect_params("Exl3MoEMethod._rank_sliced_runtime", M._rank_sliced_runtime,
              ["self", "layer", "x", "topk_ids"])
expect_params("Exl3MoEMethod._bind_rank_sliced", M._bind_rank_sliced,
              ["runtime", "layer", "plan", "scratch", "output", "x", "topk_weights", "topk_ids"])
check("_bind_rank_sliced is staticmethod",
      isinstance(inspect.getattr_static(M, "_bind_rank_sliced"), staticmethod))
expect_params("Exl3MoEMethod._apply_rank_sliced", M._apply_rank_sliced,
              ["self", "layer", "x", "topk_weights", "topk_ids"])
expect_params("Exl3MoEMethod.apply", M.apply,
              ["self", "layer", "x", "topk_weights", "topk_ids", "shared_experts",
               "shared_experts_input"])
expect_params("exl3._exl3_moe_weight_loader", exl3._exl3_moe_weight_loader,
              ["param", "loaded_weight", "weight_name", "shard_id", "expert_id", "return_success"])
expect_params("Exl3MoEParameter.map_global_expert_id", exl3.Exl3MoEParameter.map_global_expert_id,
              ["self", "expert_id"])
expect_params("exl3._load_b12x_fused_moe", exl3._load_b12x_fused_moe, [])
for attr in ("get_current_vllm_config_or_none", "_SPARKINFER_FUSED_MOE_API", "_RANK_SLICED_RUNTIMES"):
    check(f"exl3 module attr {attr}", hasattr(exl3, attr))
for attr in ("get_tensor_model_parallel_rank", "get_tensor_model_parallel_world_size"):
    check(f"vllm.model_executor.parameter.{attr} (patched during create_weights)", hasattr(vparam, attr))
src_param = inspect.getsource(vparam.BasevLLMParameter.__init__)
check("BasevLLMParameter.__init__ reads TP rank/size via module globals",
      "get_tensor_model_parallel_rank()" in src_param and "get_tensor_model_parallel_world_size()" in src_param)
expect_params("determine_expert_map", determine_expert_map,
              ["ep_size", "ep_rank", "global_num_experts", "expert_placement_strategy",
               "num_fused_shared_experts", "return_expert_mask"])
n_local, emap, _ = determine_expert_map(2, 0, 288, "linear")
check("determine_expert_map(2,0,288,linear) = experts 0..143",
      n_local == 144 and emap[:144].tolist() == list(range(144)) and (emap[144:] == -1).all().item())
check("MoEActivation.SILU value", MoEActivation.SILU.value == "silu")

api = exl3._load_b12x_fused_moe()
for attr in ("plan_weights", "prepare_weights", "Caps", "plan", "bind", "run"):
    check(f"b12x api.{attr} callable", callable(getattr(api, attr, None)))
res["api_type"] = type(api).__name__

from b12x.moe.fused_moe import _impl  # noqa: E402
from b12x.moe._shared.kernels.w4a16 import mixed_trellis  # noqa: E402

fields = {f.name for f in dataclasses.fields(mixed_trellis.MixedTrellisCompileResult)}
need = {"size_m", "fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n", "moe_block_size",
        "max_m_blocks", "blocks_per_sm", "route_ids_dtype", "direct_topk_routes"}
check("MixedTrellisCompileResult launch fields", need <= fields, sorted(need - fields))
bfields = {f.name for f in dataclasses.fields(_impl.TPMoEFP4Binding)}
check("TPMoEFP4Binding.mixed_trellis_binding", "mixed_trellis_binding" in bfields)
pfields = {f.name for f in dataclasses.fields(_impl.TPMoEScratchPlan)}
check("TPMoEScratchPlan caps/_mixed_trellis_launches", {"caps", "_mixed_trellis_launches"} <= pfields)
check("_trellis_exact_launch_token_counts(32) = 1..32",
      _impl._trellis_exact_launch_token_counts(32) == tuple(range(1, 33)))
check("_trellis_exact_launch_token_counts(16) = 1..16",
      _impl._trellis_exact_launch_token_counts(16) == tuple(range(1, 17)))
check("_trellis_exact_launch_token_counts(1024) = (1024,)",
      _impl._trellis_exact_launch_token_counts(1024) == (1024,))
check("_MIXED_TRELLIS_DIRECT_ROUTE_LIMIT == 0", getattr(_impl, "_MIXED_TRELLIS_DIRECT_ROUTE_LIMIT", None) == 0)
try:
    # informational (the harness records the tile from the bound launch);
    # 0.7.0's helper has no hidden/intermediate arguments, 0.9.1's does.
    tparams = params(_impl._projection_mixed_tile_config)
    res["tile_rule_params"] = tparams

    def tile(inter, tc):
        kw = dict(token_count=tc, direct_topk_routes=False)
        if "hidden_size" in tparams:
            kw.update(hidden_size=4096, intermediate_size=inter)
        return _impl._projection_mixed_tile_config(None, **kw)
    tiles = {(inter, tc): tile(inter, tc) for inter in (1024, 2048) for tc in (1, 9, 10, 32, 1024)}
    res["tile_rule"] = {f"I{k[0]}.m{k[1]}": list(v) for k, v in tiles.items()}
    check("tile rule K128/N128 at 10..32, K64/N256 at 1..9 and 1024 (I=1024 and 2048)",
          all(tiles[(i, 10)] == (128, 128, 128, 128) and tiles[(i, 32)] == (128, 128, 128, 128)
              and tiles[(i, 9)] == (64, 256, 64, 256) and tiles[(i, 1024)] == (64, 256, 64, 256)
              for i in (1024, 2048)))
except Exception as exc:  # noqa: BLE001
    check("tile rule", False, repr(exc))

# verbatim served-path lines the harness mirrors
src_apply_rs = inspect.getsource(M._apply_rank_sliced)
for line in ('if runtime["min_trellis_m"] <= m <= runtime["max_trellis_m"]:',
             'plan=runtime["trellis_plan"],', 'plan=runtime["prefill_plan"],',
             'output = runtime["api"].run(binding=binding)', 'return output.to(x.dtype)',
             'if runtime["prefill_plan"] is not None and m > runtime["max_trellis_m"]:'):
    check(f"_apply_rank_sliced contains: {line}", line in src_apply_rs)
src_apply = inspect.getsource(M.apply)
for line in ("x_2d = x.reshape(-1, x.shape[-1]).contiguous()",
             "ids = topk_ids.reshape(x_2d.shape[0], -1).contiguous()",
             "weights = topk_weights.reshape_as(ids).to(torch.float32).contiguous()"):
    check(f"apply contains: {line}", line in src_apply)
src_rt = inspect.getsource(M._rank_sliced_runtime)
for line in ('max_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32)',
             'block_m = _positive_env_int("VLLM_EXL3_TRELLIS_BLOCK_M", 8)',
             'prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_BLOCK_M", 64)',
             '"VLLM_EXL3_PREFILL_CAPACITY", 1024',
             'scheduler_max_batched_tokens, prefill_capacity',
             '"B12X_EXL3_BF16_EPILOGUE", "0"',
             'trellis_plan, trellis_scratch, trellis_output = _plan_with_scratch(',
             '_plan_with_scratch(max_batched_tokens, prefill_block_m)',
             '"mixed_trellis_route_id_dtypes": (topk_ids.dtype,),',
             'w4a16_block_size_m=plan_block_m,'):
    check(f"_rank_sliced_runtime contains: {line}", line in src_rt)
src_cw = inspect.getsource(M.create_weights)
for line in ("get_current_vllm_config_or_none()", "scheduler_config.max_num_batched_tokens",
             '("w13", "trellis"): 1,', '("w2", "trellis"): 0,'):
    check(f"create_weights contains: {line}", line in src_cw)
src_pm = inspect.getsource(M._prepare_projection_mixed_weights)
for line in ('source_format="exl3_trellis_mcg"', "tile_config = None",
             'trellis_rate_granularity="per_expert_projection"'):
    check(f"_prepare_projection_mixed_weights contains: {line}", line in src_pm)
check("exl3 never forwards swiglu_limit to B12x (stock routed path has no clamp)",
      "swiglu" not in inspect.getsource(exl3))

for name in ("create_weights", "process_weights_after_loading", "_prepare_projection_mixed_weights",
             "_rank_sliced_runtime", "_bind_rank_sliced", "_apply_rank_sliced", "apply"):
    res["def_lines"][f"Exl3MoEMethod.{name}"] = inspect.getsourcelines(getattr(M, name))[1]
res["def_lines"]["_load_b12x_fused_moe"] = inspect.getsourcelines(exl3._load_b12x_fused_moe)[1]
res["exl3_sha256"] = hashlib.sha256(open(exl3.__file__, "rb").read()).hexdigest()
res["torch"] = torch.__version__
res["pass"] = not res["failures"]
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(dict(engine=ENGINE, pass_=res["pass"], n=len(res["checks"]), failures=res["failures"])))
sys.exit(0 if res["pass"] else 1)
