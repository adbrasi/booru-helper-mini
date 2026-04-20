# booru-helper-mini

ComfyUI custom node pack with three separate tagger nodes:

- **WD14 Tagger** — SmilingWolf WD14 models (ONNX)
- **Camie Tagger** — `Camais03/camie-tagger-v2` (ONNX)
- **PixAI Tagger** — `pixai-labs/pixai-tagger-v0.9` (PyTorch, default) or `deepghs/pixai-tagger-v0.9-onnx` (ONNX, no auth required)

Install in `ComfyUI/custom_nodes/booru-helper-mini` and run:

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are provided by ComfyUI itself; `timm` is only required for the PixAI PyTorch backend.

## Features

- Three independent nodes — pick the model you need
- Real ONNX batching with user-defined `batch_size`
- LRU session caching (switching between WD14 variants reuses loaded sessions)
- Tuned `ort.SessionOptions` + CUDA provider options for fast inference
- Vectorized threshold selection with numpy
- PixAI: FP16 support on CUDA, optional IP (copyright/series) mapping
- Output is a list of strings (one per image in the batch)

## PixAI — HuggingFace token

The official PyTorch repo is gated. Set a token via env var before launching ComfyUI:

```bash
export HF_TOKEN="hf_..."
```

Or paste it into the `hf_token` field of the PixAI node. If you don't have access, switch the node's `backend` to `onnx` and it will use `deepghs/pixai-tagger-v0.9-onnx` instead (no auth needed).

## Models directory

Models are stored under `./models/{wd14,camie,pixai}/` inside the node folder. If your ComfyUI has a `wd14_tagger` folder registered in `folder_paths`, that one is used as base instead.
