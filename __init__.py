"""ComfyUI-ConvRot-ControlNet

Stock ComfyUI (0.33.x) loads quantized (.comfy_quant) diffusion models and
text encoders through MixedPrecisionOps, but the ControlNet and model-patch
loaders still do

    dtype = comfy.utils.weight_dtype(sd)      # -> torch.int8 for INT8 files
    Model(..., dtype=dtype, operations=comfy.ops.manual_cast)

which fails with "Only Tensors of floating point and complex dtype can
require gradients" and never binds .comfy_quant / .weight_scale tensors.

This package monkey-patches the two loader nodes so that, when a file
carries .comfy_quant metadata, they:
  * use the metadata's orig_dtype (e.g. bf16) as the compute dtype
  * build the model with the same MixedPrecisionOps the diffusion loader
    would pick (comfy.ops.pick_operations with a quant_config)
Files without quant metadata go through the untouched stock path.

No new nodes are registered.
"""
import json
import logging
import os
import sys
import types

import torch

import comfy.controlnet
import comfy.model_management
import comfy.ops
import comfy.utils
import folder_paths
import nodes

log = logging.getLogger("ComfyUI-ConvRot-ControlNet")
_TAG = "[ConvRot-ControlNet]"  # ComfyUI prints only %(message)s, so tag by hand

_PATCHED = {}  # class_name -> bool, reported in one summary line at the end


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quant_compute_dtype(sd):
    """Return the compute dtype for a quantized state dict, or None if the
    state dict carries no .comfy_quant metadata."""
    for k, v in sd.items():
        if not k.endswith(".comfy_quant"):
            continue
        try:
            conf = json.loads(v.numpy().tobytes())
        except Exception as e:  # noqa: BLE001
            log.warning("%s could not parse %s: %s", _TAG, k, e)
            conf = {}
        name = str(conf.get("orig_dtype", "torch.bfloat16")).replace("torch.", "")
        return getattr(torch, name, torch.bfloat16)
    return None


def quant_ops(compute_dtype):
    """Same selection the diffusion-model loader makes for a quantized file."""
    cfg = types.SimpleNamespace(quant_config={"mixed_ops": True})
    return comfy.ops.pick_operations(None, compute_dtype, model_config=cfg)


class _Proxy:
    """Attribute proxy: overrides win, everything else falls through."""

    def __init__(self, target, overrides):
        self._target = target
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


# --------------------------------------------------------------------------- #
# ControlNetLoader / DiffControlNetLoader  (models/controlnet)
# --------------------------------------------------------------------------- #
def _load_controlnet_path(ckpt_path, model=None):
    # mirrors comfy.controlnet.load_controlnet, plus quant detection
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    model_options = {}
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"):
        model_options["global_average_pooling"] = True

    compute_dtype = quant_compute_dtype(sd)
    if compute_dtype is not None:
        log.info("%s quantized controlnet detected (%s), compute dtype %s, using MixedPrecisionOps",
                 _TAG, os.path.basename(ckpt_path), compute_dtype)
        model_options["dtype"] = compute_dtype
        model_options["custom_operations"] = quant_ops(compute_dtype)

    cnet = comfy.controlnet.load_controlnet_state_dict(sd, model=model, model_options=model_options)
    if cnet is None:
        raise RuntimeError("ERROR: controlnet file is invalid and does not contain a valid controlnet model.")
    return cnet


def _patched_controlnet_loader(self, control_net_name):
    path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
    return (_load_controlnet_path(path),)


def _patched_diff_controlnet_loader(self, model, control_net_name):
    path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
    return (_load_controlnet_path(path, model=model),)


def _patch_mapped(class_name, method, fn):
    """Patch the class actually registered in NODE_CLASS_MAPPINGS (the one
    the executor instantiates), not a possibly separate imported copy."""
    cls = nodes.NODE_CLASS_MAPPINGS.get(class_name)
    if cls is None:
        log.warning("%s %s not in NODE_CLASS_MAPPINGS, not patched", _TAG, class_name)
        _PATCHED[class_name] = False
        return None
    setattr(cls, method, fn)
    ok = getattr(cls, method) is fn
    log.debug("%s patched %s.%s (module %s): %s", _TAG, class_name, method, cls.__module__, "OK" if ok else "FAILED")
    _PATCHED[class_name] = ok
    return cls


_patch_mapped("ControlNetLoader", "load_controlnet", _patched_controlnet_loader)
_patch_mapped("DiffControlNetLoader", "load_controlnet", _patched_diff_controlnet_loader)


# --------------------------------------------------------------------------- #
# ModelPatchLoader  (models/model_patches, e.g. Z-Image Fun ControlNet)
# --------------------------------------------------------------------------- #
# ComfyUI registers comfy_extras/*.py in sys.modules under their file path,
# not as "comfy_extras.nodes_model_patch"; importing that name would create a
# second module copy whose class is NOT the one in NODE_CLASS_MAPPINGS.
# So resolve class and module through the mapping.
_mpl_cls = nodes.NODE_CLASS_MAPPINGS.get("ModelPatchLoader")
_nmp = sys.modules.get(_mpl_cls.__module__) if _mpl_cls is not None else None
if _nmp is None:
    log.warning("%s ModelPatchLoader not found in NODE_CLASS_MAPPINGS / sys.modules, not patched", _TAG)
    _PATCHED["ModelPatchLoader"] = False
else:
    _orig_load_model_patch = _mpl_cls.load_model_patch

    def _patched_load_model_patch(self, name):
        # Per-call state, filled once the file has been read.
        utils_over = {}
        ops_over = {}

        def load_torch_file(*args, **kwargs):
            ret = comfy.utils.load_torch_file(*args, **kwargs)
            sd = ret[0] if isinstance(ret, tuple) else ret
            compute_dtype = quant_compute_dtype(sd)
            if compute_dtype is not None:
                log.info("%s quantized model patch detected (%s), compute dtype %s, using MixedPrecisionOps",
                         _TAG, name, compute_dtype)
                utils_over["weight_dtype"] = lambda sd, prefix="": compute_dtype
                ops_over["manual_cast"] = quant_ops(compute_dtype)
            return ret

        utils_over["load_torch_file"] = load_torch_file

        comfy_proxy = _Proxy(_nmp.comfy, {
            "utils": _Proxy(comfy.utils, utils_over),
            "ops": _Proxy(comfy.ops, ops_over),
        })

        # Re-bind the stock function to a globals dict where `comfy` is our
        # proxy; the original module and its globals are left untouched.
        g = dict(_nmp.__dict__)
        g["comfy"] = comfy_proxy
        fn = types.FunctionType(_orig_load_model_patch.__code__, g,
                                _orig_load_model_patch.__name__,
                                _orig_load_model_patch.__defaults__,
                                _orig_load_model_patch.__closure__)
        return fn(self, name)

    _patch_mapped("ModelPatchLoader", "load_model_patch", _patched_load_model_patch)

# One line so a broken install is diagnosable from the console.
_ok = [k for k, v in _PATCHED.items() if v]
_bad = [k for k, v in _PATCHED.items() if not v]
log.info("%s patched %d/%d loaders: %s%s",
         _TAG, len(_ok), len(_PATCHED), ", ".join(_ok) or "none",
         (" | FAILED: " + ", ".join(_bad)) if _bad else "")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
