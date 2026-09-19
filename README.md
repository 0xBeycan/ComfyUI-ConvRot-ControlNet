# ComfyUI-ConvRot-ControlNet

Load **INT8 ConvRot quantized ControlNet models** in ComfyUI with the stock loader nodes.

ComfyUI (0.33.x) can already run `int8_tensorwise` / `.comfy_quant` checkpoints through `MixedPrecisionOps` — but only for diffusion models and text encoders. The ControlNet and model-patch loaders never got that path, so an INT8 ControlNet file crashes on load. This package patches those loaders so quantized files just work, while unquantized files (bf16 / fp16 / fp8) go through the untouched stock code.

Ready-made INT8 ControlNet files: **https://huggingface.co/beycanai/ControlNet-models-INT8-ConvRot**

> **No new nodes are added.** After installing, the node list looks exactly the same. You keep using `Load ControlNet Model`, `Load ControlNet Model (diff)` and `ModelPatchLoader` — just pick the `_int8_convrot` file instead of the bf16 one. If you were expecting a new node to appear, that is why nothing shows up. Check the console for the `[ConvRot-ControlNet] patched 3/3 loaders` line to confirm it is active.

---

## Install

1. Clone into your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/0xBeycan/ComfyUI-ConvRot-ControlNet
   ```
2. Restart ComfyUI. No dependencies beyond ComfyUI itself.
3. Confirm in the console:
   ```
   [ConvRot-ControlNet] patched 3/3 loaders: ControlNetLoader, DiffControlNetLoader, ModelPatchLoader
   ```
   If it says `FAILED: ...`, the patch did not attach to that loader — open an issue with your ComfyUI version.

Put the model files where they normally go:

| File | Folder | Loader node |
|---|---|---|
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1_int8_convrot.safetensors` | `models/model_patches/` | `ModelPatchLoader` |
| `Qwen-Image-2512-Fun-Controlnet-Union-2602_int8_convrot.safetensors` | `models/controlnet/` | `Load ControlNet Model` |
| `Qwen-Image-InstantX-ControlNet-Union_int8_convrot.safetensors` | `models/controlnet/` | `Load ControlNet Model` |

Requires a ComfyUI build with native INT8 support (`comfy.quant_ops.QUANT_ALGOS["int8_tensorwise"]`, ComfyUI ≥ 0.33). Tested on 0.33.1.

---

## What you get

Tested on the three models below — each converted with the recipe further down and compared against the original bf16 at the same seed.

| Model | Loader | bf16 → INT8 file | VRAM saved | Speed | Quality (PSNR vs bf16) |
|---|---|---|---|---|---|
| Z-Image-Turbo-Fun-Controlnet-Union-2.1 | `ModelPatchLoader` | 6.71 → 3.36 GB | −3.0 GB | same | 42.0 dB |
| Qwen-Image-2512-Fun-Controlnet-Union-2602 | `Load ControlNet Model` | 3.51 → 1.82 GB | −1.6 GB | same | 36.0 dB |
| Qwen-Image-InstantX-ControlNet-Union | `Load ControlNet Model` | 3.54 → 1.83 GB | −1.6 GB | same | 44.8 dB |

Test hardware: RTX 5090, ComfyUI 0.33.1, PyTorch 2.10.0+cu130.

**The gain is VRAM and disk, not speed.** On the 5090 generation time was identical to bf16. Older cards (30/40 series) may see a speed-up from the INT8 kernel, but that has not been tested — treat any such number as unverified.

---

## Why stock ComfyUI fails

Two independent breaks, both in `comfy/controlnet.py` (`controlnet_load_state_dict`) and `comfy_extras/nodes_model_patch.py` (`ModelPatchLoader`):

1. **dtype.** Both loaders pick the architecture dtype with `comfy.utils.weight_dtype(sd)`. For an INT8 checkpoint that returns `torch.int8`, and building the module graph with an int8 dtype dies immediately:
   ```
   RuntimeError: Only Tensors of floating point and complex dtype can require gradients
   ```
2. **ops.** These paths have no `model_config.quant_config` and never pass `custom_operations`, so even with a float dtype the `.comfy_quant` / `.weight_scale` tensors are ignored and no quantized tensor is bound to the layer.

The patch fixes both: when a file carries `.comfy_quant` metadata it uses the metadata's `orig_dtype` (normally bf16) as the compute dtype and passes the same `MixedPrecisionOps` the diffusion loader would pick. Files without metadata fall through to the stock path, so this is a drop-in — nothing changes for your existing ControlNets.

Patched loaders: `ControlNetLoader`, `DiffControlNetLoader`, `ModelPatchLoader`.

### Technical note: why the patch goes through `NODE_CLASS_MAPPINGS`

Replacing the method on the class you `import` is not enough. `NODE_CLASS_MAPPINGS` may hold a *different* class object than the one you get from an import — `comfy_extras/*.py` files are registered in `sys.modules` under their file path, so `import comfy_extras.nodes_model_patch` creates a second copy of the module whose class is not the one the executor instantiates. Our first attempt loaded fine and silently did nothing for exactly this reason.

The package therefore resolves each class *through* `nodes.NODE_CLASS_MAPPINGS`, patches that object, and verifies the attribute stuck. For `ModelPatchLoader` it goes one step further: it re-binds the stock `load_model_patch` code to a globals dict where `comfy.utils.weight_dtype` / `comfy.ops.manual_cast` are proxied per call, leaving the original module untouched. If you fork this, keep that mechanism.

---

## Conversion recipe

Tool: [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)

```bash
pip install convert-to-quant
```

```bash
ctq -i <input>.safetensors -o <output>_int8_convrot.safetensors \
    --int8 --convrot --convrot-group-size 256 --scaling_mode row \
    --exclude-layers "<regex>" \
    --comfy_quant --save-quant-metadata --simple --low-memory --device cuda
```

**`--scaling_mode row` is mandatory.** Without it the tool emits a tensor-wise scalar scale, no `per_row` metadata is written, LoRAs break on top of the model and the output visibly softens.

### Exclusion lists per model

| Model | `--exclude-layers` |
|---|---|
| Z-Image-Turbo-Fun-Controlnet-Union-2.1 | `control_all_x_embedder` |
| Qwen-Image-2512-Fun-Controlnet-Union-2602 | `control_blocks\.\d+\.(before\|after)_proj\|control_img_in` |
| Qwen-Image-InstantX-ControlNet-Union | `controlnet_blocks\|controlnet_x_embedder\|img_in\|time_text_embed` |

Copy-paste form:

```bash
# Z-Image Union 2.1 (in_features 132, does not divide into the 256 group)
--exclude-layers "control_all_x_embedder"

# Qwen-Image 2512 Fun
--exclude-layers "control_blocks\.\d+\.(before|after)_proj|control_img_in"

# Qwen-Image InstantX
--exclude-layers "controlnet_blocks|controlnet_x_embedder|img_in|time_text_embed"
```

Why these layers stay in bf16: the zero-initialised / small-amplitude injection layers (`after_proj`, `controlnet_blocks`, …) carry 4–8× smaller magnitudes than the regular transformer layers. Quantizing them with the same settings buries the signal in quantization noise and can silently mute the conditioning — the model loads, runs, and just stops following the control image. They are small, so leaving them in bf16 costs almost nothing.

---

## License

MIT. See `LICENSE`.
