import transformers

def replace_llama_rmsnorm_with_fused_rmsnorm():
    try:
        from apex.normalization import FusedRMSNorm
        from functools import partial
        LlamaRMSNorm = partial(FusedRMSNorm, eps=1e-6)   # noqa
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        print("Discovered apex.normalization.FusedRMSNorm - will use it instead of LlamaRMSNorm")
    except ImportError:
        # using the normal LlamaRMSNorm
        pass
    except Exception:
        print("discovered apex but it failed to load, falling back to LlamaRMSNorm")
        pass
