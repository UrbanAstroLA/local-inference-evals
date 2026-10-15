import sys; sys.path.insert(0, "/replay")
import engine_replay as E
s = E.make_scheduler()
km = s.kv_cache_manager
c = km.coordinator
print("block sizes", s._replay_block_sizes, type(c).__name__, "eagle", sorted(c.eagle_group_ids),
      "partial", c.enable_partial_hash_hits, "align_tokens", c._cache_hit_alignment_tokens)
print([type(m).__name__ for m in c.single_type_managers], [m.use_eagle for m in c.single_type_managers])
print([ (type(g.spec).__name__, g.group_ids, g.use_eagle) for g in c.attention_groups])
print("need split", s.need_mamba_block_aligned_split, "lookahead", s.num_lookahead_tokens, "use_eagle", s.use_eagle)
print(E.placement_summary(s))
