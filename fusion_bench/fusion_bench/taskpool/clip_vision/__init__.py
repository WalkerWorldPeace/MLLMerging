# flake8: noqa F401
# The MoE / smile / sparse_wemoe taskpools depend on transformers internals
# (``CLIPVisionTransformer``) that were removed in transformers 5.x; guard the
# imports so that the standard CLIPVisionModelTaskPool remains usable under
# newer transformers versions.
try:
    from .clip_rankone_moe_taskpool import RankoneMoECLIPVisionModelTaskPool
except ImportError as _e:
    RankoneMoECLIPVisionModelTaskPool = None  # type: ignore
try:
    from .clip_smile_taskpool import SmileCLIPVisionModelTaskPool
except ImportError as _e:
    SmileCLIPVisionModelTaskPool = None  # type: ignore
try:
    from .clip_sparse_wemoe_taskpool import SparseWEMoECLIPVisionModelTaskPool
except ImportError as _e:
    SparseWEMoECLIPVisionModelTaskPool = None  # type: ignore
from .taskpool import CLIPVisionModelTaskPool
