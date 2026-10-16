# Sources for RANKING.md

Every local file relied on, with sha256 at the time of reading (2026-10-15). Paths are absolute. The engine source trees
under the session scratchpad (`.../scratchpad/kdcp/v070`, `v091`) are extractions of the release images made earlier
in this investigation (source review "Sources"): `v070` = image sha256:48e254d9… (vLLM 0.1.dev20051+g487ecf187, B12x
fe054789), `v091` = image sha256:a9e985c0… (B12x 7fcc094e). Scratchpad copies are session-local; the image digests are
the durable reference.

## Records and write-ups (read only)

| file | used for | sha256 |
|---|---|---|
| `<runs>/defect-repros/INDEX.md` | P1/P2/P3 and D1-D6 status, image digests, two-rank P2 figures | `b525a7dbae6342060ea2f813cf4aa095b4c2957e1e7322fa10f08a1d4d54c6a3` |
| `<runs>/defect-repros/D2-kpool-tail-ring/README.md` | D2 mechanism (file:line), 0.7.0 DCP2 100%, server verification | `81735e408347feeecb162c7ef8e65730e269bac69b948fbe629b97dcb2bd6d28` |
| `<runs>/defect-repros/D2-kpool-tail-ring/two-rank/summary.md` | per-cell wrong-pool shares and shadow classes | `a902c5036738cb770ef633b35d8d4230762518cebfcb6a607b9f1045bf2f6358` |
| `<runs>/defect-repros/D2-kpool-tail-ring/two-rank/two_rank_tailring.py` | harness copied (with overrides) into p2-exposure/; acceptance model (iid conditional) | `dc0c85a56459db9b1c6284c4be46b8f59d84f14f0aa2ddf153a374a2e0cefc06` |
| `<runs>/defect-repros/D2-kpool-tail-ring/run_two_rank.sh` | container invocation reused | `588ca9695b5dfe120b6746787597801d2fa2ca8cfa1b410c48b989fe7ad0a2de` |
| `<runs>/defect-repros/D2-kpool-tail-ring/expected/two-rank/v070-a0.85.json` | per-seed shadow classes, shipped 0.7.0 | `94767f48b38b2532c2ebf182ccd5bf90e74c00a2f48650832362b955759c4adb` |
| `<runs>/defect-repros/D2-kpool-tail-ring/expected/two-rank/v070-a0.6.log` | harness timings | `5ba7db6c5bdb972ca3908a5fbc9f83031dccececbc5931cf945df76372e61db6` |
| `<runs>/defect-repros/D2-kpool-tail-ring/expected/two-rank/v091-a0.6.log` | harness timings | `8aeddd05dd1a036af5c5dcf5d712503058211cb9e42a53bc96aee4bff47715de` |
| `<runs>/source-review-lc/SOURCE-REVIEW.md` | route thresholds, geometry, draft-slot sharing (s4), KDA (s5), sampler (s6), 0.7.0 merge | `e69df4441890ea856eaa0eaf2149905780f82cf772c4d652f7a0a22ac044af0f` |
| `<runs>/k-selection/DIAGNOSIS-run1.md` | atomic-slot mechanism, half-pool consumer analysis, claim-group model | `759a0312698e2301751cb3540f45776dd2bbfea7b8136e649ada1b62ec9edd5e` |
| `<runs>/k-selection/diagnosis-run1/crosscheck_col511.out` | column-511 rank<480 decomposition by family | `36d074dff03e84b498ea6f56286800f20d808188396bac0001deaa99d0cce4b5` |
| `<runs>/k-selection/run-20261014T005710Z/REPORT.txt` | K rerun per-cell column-511 divergence, relaunch control | `6d5376bbf59da26c4742763cdc927e08ad6c94e643475ba3eff7d621b3f69e84` |
| `<runs>/k-selection/run-20261014T005710Z/v070/summary-v070.json` | per-family column-511 divergence counts | `ff4750943a60ee69fd02bcbb9df99c50a6f25b965abdb2826fff4907aabe75b2` |
| `<runs>/k-selection/run-20261014T005710Z/v070/results-v070.jsonl` | record-level persp / relaunch flags | `a9e466990b2b7a46eb4c43ee89649f504b279331d3d3652d88baf4b4f13ead36` |
| `<runs>/rep070/PROPOSAL.md` | replication cost (22.97 / 26.45 / 35.0) and power (O1/O2) | `85d59ed5238487e2a8113785d108b53b5ef585ad79d5983d2b19925f0f1f9c59` |
| `<runs>/replication/summary.md` | decode throughput at 8 in flight (gen_tps_median ~291) | `ffb3c20385d92743fd4508effb1f4ea70343b3c0fa25fa989785201987883fda` |
| `<runs>/probe-proposal/PROPOSAL.md` | B0-ring4 build recipe, oracle-draft bracket, served acceptance lengths | `6acb37d78a25333e247e57c7484214c659fa51920c8963a22fcdf131f0a51bf6` |
| `<runs>/first-contrast/PROPOSAL.md` | direction rule, ring-4 out of scope | `863aaa7e3bf2b33f0a047455629ea2d57ed7c2171091e201939859ca0eac8391` |
| `<runs>/first-contrast/retry/cost/cost_fcr.out` | measured-rate costing (A0/B central and pessimistic, early-stop costs, 2.84 tokens/step) | `a441093796916980fbb707e533d597ca4dd8ce568d267a4a042ce9e41dc1ada4` |
| `<runs>/first-contrast/retry/costs.jsonl` | fcr1 2.9317 GPU-hours | `bfdb3010110fbf679ea3d4e7a7a613d293bd3c7b7610469976cc30ac6b177c8b` |
| `<runs>/first-contrast/retry/run-fcr1/fcr_result.txt` | fcr1 NO READING, delta 0.106 / 0.200, H5d ~70% gross, selection divergence | `3c565e0fc89057429295f963f9b7028633814069f701590c584d852ab9bfd906` |
| `<this-repo>/investigations/2026-10-glm53-completion/PLAN.md` | amendments 2-7 (steps 2-6), 19f, 20-20d, 21-21k, 22-22k | `cb206095890104dce16845a885dd515e376713da5b5c7c587f71d5ec84ffe38a` |
| `<this-repo>/investigations/2026-10-glm53-completion/README.md` | step 6 table and verdict, open-question table | `6bac53b9ed3a1492eba49be1c0a094ae7357eb97c5bfe485f53f3b95c309bcb4` |
| `<this-repo>/investigations/2026-10-glm53-completion/costs.jsonl` | GPQA pass costs (B1 6.33, B0 5.866), guarded sessions (0.62-0.99), step 5 1.615 | `e77298fee208089afc86ad31334593d4f72cf2bbcf7e029da1f833fc4422aa4c` |
| `<this-repo>/FINDINGS.md` | item 4n, 4k context | `25798df79be76a7a47afbe89ce1fd9523c9a222db1190bd963651d0715049976` |
| `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+ps.server.log` | 0.7.0 GPQA passes 2-3: acceptance, concurrency, KV peak (counts only) | `30e9e9da2a19b4c2d1c54433cccbdd78a11fdbaf89fbce39881946f2a75bcd6f` |
| `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log` | 0.9.1 GPQA passes 2-3: acceptance, concurrency | `0e05caefa4ab0538957cbf2d90acf296a11bf5e5204e6f0d9fb2b373d35b622e` |
| `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log` | 0.9.1 GPQA pass 1: acceptance, concurrency | `0ef6aaff01c06204755a2647c9dadadff65b46b2e4086d0de6b8fd09d7ef8b61` |

## Engine sources (read; lines cited in RANKING.md)

| file | lines relied on | sha256 |
|---|---|---|
| `<scratch>/kdcp/v070/vllm/models/glm5next/nvidia/attention.py` | 192 (tail `block_size=self._index_kpool`, ring 4) | `7d42294c63ea52a7845a6ad0289ae3596098f5aee857718cfed34d6dcf0418f2` |
| `<scratch>/kdcp/v091/vllm/models/glm5next/nvidia/attention.py` | 191-203 (kpool-spec-ring sizing: ring 8 at K=3, 16 at K=5) | `c3b368160f00a60c192873ba127bb1c5ec8a98bbc7e1ee7a887152f51179db53` |
| `<scratch>/kdcp/v070/vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 575-580 (ring slot `pos % POOL_SIZE`); diff against v091 (RING parameter, seed stride) | `52959ea33234db6e7cfa94289dd263a15add3be4d2107285418c830b5c19e472` |
| `<scratch>/kdcp/v091/vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 578-580 (ring holds a pool plus every draft) | `7fb8eab0d71c2c576f995cc91ffc79c5055269b9bed3b094247f36bc503b241d` |
| `<scratch>/kdcp/v070/vllm/model_executor/layers/sparse_attn_indexer.py` | 1139-1218 (`_merge_b12x_dcp_topk`: all-gather, 1,024 candidates), 1207 (`run_row_topk`) | `d5091557f9b12d968b323a6778838197464cf9e9c4959239de834648c001984d` |
| `<scratch>/kdcp/v070/vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 63-67 (B12x kpool enable), 881-887 (route condition), 924 (merge call), 1009 (truncation to 511) | `e7935f2a9b8ecbc75410defff571d9940ad2abf511286bf0936d0c324936df39` |
| `<scratch>/kdcp/v070/vllm/envs.py` | 61-65, 1087-1100 (DCP top-k env defaults; owner exchange not on the kpool path) | `146781ccbd1f2f616e9c0459cf3b4643f2ed0be90cb33e5c154e134bcf1f8dfa` |
| `<scratch>/kdcp/v070/b12x/b12x/attention/dsa_indexer/tiled_topk.py` | 893-896 (output slot via `_smem_xadd`), 1043-1045 (tie fill counter) | `d0d43d9f02ff6c683d0c0c4efcfcb2324749c9dc9998f0d46b3952bbbb6e8d6d` |

## Images used for the CPU runs (`p2-exposure/run_p2_exposure.sh`)

| image | digest |
|---|---|
| B1 + guard + hook (`local/glm53-v0.9.1-b1-forcehook-mg2`) | `sha256:ac05230bd94b3114cb6fb202b06402f9e2d45a61b20f041f314e045f967e97ee` |
| stock 0.7.0 | `sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629` |
| stock 0.9.1 | `sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07` |

## Files written in this folder

(SOURCES.md itself excluded.)

| file | sha256 |
|---|---|
| `./p2-exposure/out/b1-r4-k3-a0.6.json` | `53ae8fa0fd6a11c3da5f0881079ed0467ad8fce255a4d7a3fd670afdaa4cdd90` |
| `./p2-exposure/out/b1-r4-k3-a0.6.log` | `c788be53f52224cecd09b5ef667c58f8d9908636029d05865d0eee69973a5cba` |
| `./p2-exposure/out/b1-r4-k3-a0.77.json` | `054c0dbc1407050c1b9d2761fbcb6077171c92ce4324e026743f080a87153c4c` |
| `./p2-exposure/out/b1-r4-k3-a0.77.log` | `6cc2e884faf01aefdc46b989c556e86d72bf445cb21c57f9f355556705357950` |
| `./p2-exposure/out/b1-r4-k3-a0.85.json` | `49720767ca9c70795d33d68d066547532a520a72604651d956d7c7204797c0dc` |
| `./p2-exposure/out/b1-r4-k3-a0.85.log` | `dc525f0e956799b7aa45d1b1f9b94e627c75aa27280cfcfdd3bddc0badcb886c` |
| `./p2-exposure/out/b1-r4-k5-a0.72.json` | `9c54e1488d888e5a65aa7dbb649b30c3f1a7eeed7c62f66a3bcbdbe569c6d0f2` |
| `./p2-exposure/out/b1-r4-k5-a0.72.log` | `25d71dbd02dbee13051ff0329f34b230443e31cab4399ed9caefafbc25d287e2` |
| `./p2-exposure/out/b1-r8-k3-a0.77.json` | `da1d97d2e46079c5b86235a5c395ea2cf0d18e9d9551dcb7a3184fffd4f6ba29` |
| `./p2-exposure/out/b1-r8-k3-a0.77.log` | `ed70e54440c0317d578b4fca1daa901c83cee30ec8ae36d24b920c9adba8d787` |
| `./p2-exposure/out/b1-r8-k5-a0.72.json` | `11109aabc45e28a4f9d7bdc0d4c29d3aedc4b9e173e7c89dc6ec5622731f733d` |
| `./p2-exposure/out/b1-r8-k5-a0.72.log` | `819a60780362f0a60bd2eab0301674693180a0687536a2295f3c94e69a3f6fc2` |
| `./p2-exposure/out/v070-a0.72.json` | `5c99f80dd5afbf46804e8634a9fd78cb48f7205d7fa62e1949c0bee172f2d4c6` |
| `./p2-exposure/out/v070-a0.72.log` | `2b1a2ec64a95825b86f301052befcfb05d59b67328cc6af2f1bffdfd566d3203` |
| `./p2-exposure/out/v091-dcp2-r8-k3-a0.77.json` | `6cbed58f67ab84c1a9b8461dbfea5654a180acc1abf2a74e806082e3aee7720c` |
| `./p2-exposure/out/v091-dcp2-r8-k3-a0.77.log` | `32b34b07b194369e6c16000da90f25b35bf6888dc6342908f1d152e00ff60443` |
| `./p2-exposure/p2_exposure.py` | `e8e0a0738489b0e93a9be326837ef46f0d7ef066acfc9a9eaea9cb0ab7e05aa5` |
| `./p2-exposure/run_p2_exposure.sh` | `af1fe5e6ce944466868ad0d9101defcfb1164e49054a23a299a44ab1a8e3fae0` |
| `./p2-exposure/summarize.py` | `6b40bf3fed1faa36b8ca997617e699162de22f2f1f98a30c4a4139cb48eaa404` |
| `./p2-exposure/summary.md` | `33c87cb0d5e578f8720583b896cf4327a88bdf6ad09686e821cec8e363c57c24` |
| `./power_long_finish.out` | `adbde9606f1128dcbef658f3468bb7675489fd6b46452bb68e2adab9fbee86bb` |
| `./power_long_finish.py` | `9bfc8eb084fc71a9de60fd22544b0ed238445d7a9973473d1f3aa37e947028e8` |
| `./RANKING.md` | `18f925ac8996054c48162367ec8f5a8c02d36a50bc8677b90a3d6e8e35d38544` |
| `./served_spec_metrics.out` | `926338a057d9cef5e6a06f73a9e41eb6c86caa09f8b73f5b6247033950b634fd` |
| `./served_spec_metrics.py` | `95f0015aa2fe32dcfc482a99452681ea00b000e053c3db3a1e2a9a45fca4bc76` |
