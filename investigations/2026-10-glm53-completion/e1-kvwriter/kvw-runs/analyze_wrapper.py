# Loader wrapper (post-GPU, before any analysis output existed): the result files were written by this investigation's own
# kvw_gpu_compare.py and contain torch.torch_version.TorchVersion, which torch>=2.6 weights_only loading rejects. This allows
# that one type and then runs the frozen kvw_analyze.py unchanged (its sha256 is in freeze/pre-gpu.sha256).
import sys, runpy, torch
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
sys.argv = ["/w/kvw_analyze.py"] + sys.argv[1:]
runpy.run_path("/w/kvw_analyze.py", run_name="__main__")
