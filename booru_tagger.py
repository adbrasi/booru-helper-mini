import csv
import gc
import json
import os
import shutil
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple

import comfy.utils
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths

try:
    import torch
except Exception:
    torch = None

try:
    import timm
    import torchvision.transforms as T
except Exception:
    timm = None
    T = None


WD14_MODELS = [
    "wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-convnext-tagger",
    "wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger",
]
DEFAULT_WD14_MODEL = "wd-v1-4-moat-tagger-v2"
DEFAULT_CAMIE_REPO = "Camais03/camie-tagger-v2"
DEFAULT_PIXAI_TORCH_REPO = "pixai-labs/pixai-tagger-v0.9"
DEFAULT_PIXAI_ONNX_REPO = "deepghs/pixai-tagger-v0.9-onnx"

if "wd14_tagger" in folder_paths.folder_names_and_paths:
    BASE_MODELS_DIR = folder_paths.get_folder_paths("wd14_tagger")[0]
else:
    BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

WD14_DIR = os.path.join(BASE_MODELS_DIR, "wd14")
CAMIE_DIR = os.path.join(BASE_MODELS_DIR, "camie")
PIXAI_DIR = os.path.join(BASE_MODELS_DIR, "pixai")
for _d in (WD14_DIR, CAMIE_DIR, PIXAI_DIR):
    os.makedirs(_d, exist_ok=True)


def _cleanup_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class _LRU(OrderedDict):
    def __init__(self, maxsize: int = 2):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            _, evicted = self.popitem(last=False)
            del evicted
            _cleanup_memory()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


WD14_CACHE: _LRU = _LRU(maxsize=2)
CAMIE_CACHE: _LRU = _LRU(maxsize=1)
PIXAI_CACHE: _LRU = _LRU(maxsize=1)


def _session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def _providers() -> list:
    available = ort.get_available_providers()
    providers: list = []
    if "CUDAExecutionProvider" in available:
        providers.append((
            "CUDAExecutionProvider",
            {
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
                "cudnn_conv_use_max_workspace": "1",
            },
        ))
    if "ROCMExecutionProvider" in available:
        providers.append("ROCMExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _make_session(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, sess_options=_session_options(), providers=_providers())


def _resolve_hf_token(token: str) -> Optional[str]:
    token = (token or "").strip()
    if token:
        return token
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def _tensor_to_pil_batch(image_tensor) -> List[Image.Image]:
    if hasattr(image_tensor, "detach"):
        arr = image_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(image_tensor)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _iterate_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _exclude_set(exclude_tags: str) -> set:
    return {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}


def _to_rgb(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.split()[-1])
        return base
    if img.mode == "P" and "transparency" in img.info:
        return _to_rgb(img.convert("RGBA"), bg)
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _format(tags: List[str], exclude: set, separator: str, trailing_comma: bool, replace_underscore: bool) -> str:
    out = []
    for tag in tags:
        t = tag.replace("_", " ") if replace_underscore else tag
        if t.lower() in exclude:
            continue
        out.append(t)
    s = separator.join(out)
    if trailing_comma and s:
        s += ","
    return s


# ---------- WD14 ----------

def _preprocess_wd14(img: Image.Image, size: int) -> np.ndarray:
    img = _to_rgb(img, (255, 255, 255))
    ratio = float(size) / max(img.size)
    new_size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    arr = np.asarray(square, dtype=np.float32)
    return arr[:, :, ::-1].copy()  # RGB -> BGR


def _ensure_wd14_assets(model_name: str, force_download: bool) -> Tuple[str, str]:
    onnx_path = os.path.join(WD14_DIR, f"{model_name}.onnx")
    csv_path = os.path.join(WD14_DIR, f"{model_name}.csv")
    if not force_download and os.path.exists(onnx_path) and os.path.exists(csv_path):
        return onnx_path, csv_path

    if model_name not in WD14_MODELS:
        raise FileNotFoundError(f"Unknown WD14 model: {model_name}")

    repo_id = f"SmilingWolf/{model_name}"
    cache_dir = os.path.join(WD14_DIR, "_downloads", model_name)
    os.makedirs(cache_dir, exist_ok=True)
    src_onnx = hf_hub_download(repo_id=repo_id, filename="model.onnx", local_dir=cache_dir, force_download=force_download)
    src_csv = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", local_dir=cache_dir, force_download=force_download)
    shutil.copy2(src_onnx, onnx_path)
    shutil.copy2(src_csv, csv_path)
    return onnx_path, csv_path


def _load_wd14_tags(csv_path: str):
    tags: List[str] = []
    rating_idx: List[int] = []
    general_idx: List[int] = []
    character_idx: List[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            tags.append(row[1])
            category = row[2]
            if category == "9":
                rating_idx.append(i)
            elif category == "0":
                general_idx.append(i)
            elif category == "4":
                character_idx.append(i)
    return (
        np.asarray(tags, dtype=object),
        np.asarray(rating_idx, dtype=np.int64),
        np.asarray(general_idx, dtype=np.int64),
        np.asarray(character_idx, dtype=np.int64),
    )


def _get_wd14(model_name: str, force_download: bool):
    key = model_name
    if not force_download and key in WD14_CACHE:
        return WD14_CACHE[key]

    onnx_path, csv_path = _ensure_wd14_assets(model_name, force_download)
    session = _make_session(onnx_path)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    shape = input_info.shape  # WD14 ONNX is NHWC: [batch, H, W, C]
    size = 448
    for dim in (shape[1], shape[2]):
        if isinstance(dim, int) and dim > 0:
            size = int(dim)
            break
    output_name = session.get_outputs()[0].name
    tags_arr, rating_idx, general_idx, character_idx = _load_wd14_tags(csv_path)
    entry = (session, input_name, output_name, size, tags_arr, rating_idx, general_idx, character_idx)
    WD14_CACHE[key] = entry
    return entry


def _run_wd14(images, model_name, batch_size, threshold, character_threshold, include_rating, force_download):
    session, input_name, output_name, size, tags_arr, rating_idx, general_idx, character_idx = _get_wd14(
        model_name, force_download
    )
    out: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))
    for batch in _iterate_batches(images, batch_size):
        arr = np.stack([_preprocess_wd14(img, size) for img in batch]).astype(np.float32)
        probs = session.run([output_name], {input_name: arr})[0]
        for row in probs:
            picks: List[str] = []
            if include_rating and rating_idx.size:
                picks.append(str(tags_arr[rating_idx[int(np.argmax(row[rating_idx]))]]))
            char_sel = character_idx[row[character_idx] >= character_threshold]
            picks.extend(tags_arr[char_sel].tolist())
            gen_sel = general_idx[row[general_idx] >= threshold]
            picks.extend(tags_arr[gen_sel].tolist())
            out.append(picks)
            pbar.update(1)
    return out


# ---------- Camie ----------

def _preprocess_camie(img: Image.Image, size: int) -> np.ndarray:
    img = _to_rgb(img, (124, 116, 104))
    w, h = img.size
    if w >= h:
        nw = size
        nh = max(1, int(size * h / w))
    else:
        nh = size
        nw = max(1, int(size * w / h))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (size, size), (124, 116, 104))
    padded.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    arr = np.asarray(padded, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1).copy()


def _ensure_camie_assets(repo_id: str, force_download: bool) -> Tuple[str, str]:
    model_dir = os.path.join(CAMIE_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    onnx_name = "camie-tagger-v2.onnx"
    meta_name = "camie-tagger-v2-metadata.json"
    onnx_path = os.path.join(model_dir, onnx_name)
    meta_path = os.path.join(model_dir, meta_name)
    if force_download or not os.path.exists(onnx_path):
        hf_hub_download(repo_id=repo_id, filename=onnx_name, local_dir=model_dir, force_download=force_download)
    if force_download or not os.path.exists(meta_path):
        hf_hub_download(repo_id=repo_id, filename=meta_name, local_dir=model_dir, force_download=force_download)
    return onnx_path, meta_path


def _get_camie(repo_id: str, force_download: bool):
    if not force_download and repo_id in CAMIE_CACHE:
        return CAMIE_CACHE[repo_id]

    onnx_path, meta_path = _ensure_camie_assets(repo_id, force_download)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    mapping = metadata["dataset_info"]["tag_mapping"]
    idx_to_tag = {int(k): v for k, v in mapping["idx_to_tag"].items()}
    tag_to_category = mapping["tag_to_category"]
    img_size = int(metadata.get("model_info", {}).get("img_size", 512))

    num_tags = max(idx_to_tag.keys()) + 1
    tags_arr = np.empty(num_tags, dtype=object)
    cats_arr = np.empty(num_tags, dtype=object)
    for idx, tag in idx_to_tag.items():
        tags_arr[idx] = tag
        cats_arr[idx] = tag_to_category.get(tag, "general")

    is_character = (cats_arr == "character")
    is_rating = (cats_arr == "rating")

    session = _make_session(onnx_path)
    input_name = session.get_inputs()[0].name
    entry = (session, input_name, img_size, tags_arr, is_character, is_rating)
    CAMIE_CACHE[repo_id] = entry
    return entry


def _run_camie(images, repo_id, batch_size, general_threshold, character_threshold, force_download):
    session, input_name, img_size, tags_arr, is_character, is_rating = _get_camie(repo_id, force_download)
    out: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))
    for batch in _iterate_batches(images, batch_size):
        arr = np.stack([_preprocess_camie(img, img_size) for img in batch]).astype(np.float32)
        raw = session.run(None, {input_name: arr})
        logits = raw[1] if len(raw) >= 2 else raw[0]
        probs = _sigmoid(np.asarray(logits))
        for row in probs:
            char_mask = is_character & (row >= character_threshold)
            gen_mask = (~is_character) & (~is_rating) & (row >= general_threshold)
            picks = tags_arr[char_mask].tolist() + tags_arr[gen_mask].tolist()
            out.append(picks)
            pbar.update(1)
    return out


# ---------- PixAI (PyTorch default / ONNX optional) ----------

def _preprocess_pixai(img: Image.Image) -> np.ndarray:
    img = _to_rgb(img, (255, 255, 255))
    img = img.resize((448, 448), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1).copy()


def _ensure_pixai_torch_assets(repo_id: str, token: Optional[str], force_download: bool):
    model_dir = os.path.join(PIXAI_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    files = {
        "weights": "model_v0.9.pth",
        "tags": "tags_v0.9_13k.json",
        "ip_map": "char_ip_map.json",
    }
    paths = {}
    for key, name in files.items():
        path = os.path.join(model_dir, name)
        if force_download or not os.path.exists(path):
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=model_dir, force_download=force_download, token=token)
        paths[key] = path
    return paths["weights"], paths["tags"], paths["ip_map"]


def _ensure_pixai_onnx_assets(repo_id: str, token: Optional[str], force_download: bool):
    model_dir = os.path.join(PIXAI_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")
    tags_path = os.path.join(model_dir, "selected_tags.csv")
    if force_download or not os.path.exists(onnx_path):
        hf_hub_download(repo_id=repo_id, filename="model.onnx", local_dir=model_dir, force_download=force_download, token=token)
    if force_download or not os.path.exists(tags_path):
        hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", local_dir=model_dir, force_download=force_download, token=token)
    return onnx_path, tags_path


def _load_pixai_torch_maps(tags_path: str, ip_path: str):
    with open(tags_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tag_map = payload["tag_map"]
    split = payload["tag_split"]
    gen_count = int(split["gen_tag_count"])
    char_count = int(split["character_tag_count"])

    total = gen_count + char_count
    tags_arr = np.empty(total, dtype=object)
    for tag, idx in tag_map.items():
        idx_i = int(idx)
        if 0 <= idx_i < total:
            tags_arr[idx_i] = tag

    is_character = np.zeros(total, dtype=bool)
    is_character[gen_count:gen_count + char_count] = True

    with open(ip_path, "r", encoding="utf-8") as f:
        ip_map = json.load(f)
    return tags_arr, is_character, ip_map


def _load_pixai_onnx_maps(tags_path: str):
    tags: List[str] = []
    categories: List[str] = []
    ips: List[List[str]] = []
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags.append(row.get("name", ""))
            categories.append(str(row.get("category", "0")))
            ips_raw = row.get("ips", "") or ""
            ips.append([s.strip() for s in ips_raw.split(",") if s.strip()])
    tags_arr = np.asarray(tags, dtype=object)
    cat_arr = np.asarray(categories, dtype=object)
    is_character = (cat_arr == "4")
    ip_map = {tags[i]: ips[i] for i in range(len(tags)) if ips[i]}
    return tags_arr, is_character, ip_map


def _build_pixai_torch(weights_path: str, num_classes: int, device: str, fp16: bool):
    if torch is None or timm is None:
        raise RuntimeError("PixAI pytorch backend requires torch and timm. Switch to onnx backend or install deps.")
    encoder = timm.create_model("hf_hub:SmilingWolf/wd-eva02-large-tagger-v3", pretrained=False)
    encoder.reset_classifier(0)
    feat_dim = int(getattr(encoder, "num_features", 1024))
    head = torch.nn.Linear(feat_dim, num_classes)
    model = torch.nn.Sequential(encoder, head)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    if fp16 and device.startswith("cuda"):
        model.half()
    return model


def _pixai_device(choice: str) -> str:
    if torch is None:
        return "cpu"
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _get_pixai(backend: str, repo_id: str, token: Optional[str], device_choice: str, fp16: bool, force_download: bool):
    key = f"{backend}:{repo_id}:{device_choice}:{fp16}"
    if not force_download and key in PIXAI_CACHE:
        return PIXAI_CACHE[key]

    if backend == "pytorch":
        weights, tags, ip_path = _ensure_pixai_torch_assets(repo_id, token, force_download)
        tags_arr, is_character, ip_map = _load_pixai_torch_maps(tags, ip_path)
        device = _pixai_device(device_choice)
        model = _build_pixai_torch(weights, len(tags_arr), device, fp16)
        entry = ("pytorch", model, device, fp16, tags_arr, is_character, ip_map)
    else:
        onnx_path, tags_path = _ensure_pixai_onnx_assets(repo_id, token, force_download)
        tags_arr, is_character, ip_map = _load_pixai_onnx_maps(tags_path)
        session = _make_session(onnx_path)
        input_name = session.get_inputs()[0].name
        entry = ("onnx", session, input_name, tags_arr, is_character, ip_map)

    PIXAI_CACHE[key] = entry
    return entry


def _run_pixai(images, backend, repo_id, batch_size, general_threshold, character_threshold,
               include_ip, device_choice, fp16, token, force_download):
    entry = _get_pixai(backend, repo_id, token, device_choice, fp16, force_download)
    out: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))

    if entry[0] == "pytorch":
        _, model, device, model_fp16, tags_arr, is_character, ip_map = entry
        for batch in _iterate_batches(images, batch_size):
            stacked = np.stack([_preprocess_pixai(img) for img in batch]).astype(np.float32)
            tensor = torch.from_numpy(stacked)
            if device.startswith("cuda"):
                tensor = tensor.pin_memory().to(device, non_blocking=True)
                if model_fp16:
                    tensor = tensor.half()
            else:
                tensor = tensor.to(device)
            with torch.inference_mode():
                logits = model(tensor)
                probs = torch.sigmoid(logits).float().cpu().numpy()
            for row in probs:
                _collect_pixai(row, tags_arr, is_character, ip_map, general_threshold, character_threshold, include_ip, out)
                pbar.update(1)
    else:
        _, session, input_name, tags_arr, is_character, ip_map = entry
        for batch in _iterate_batches(images, batch_size):
            arr = np.stack([_preprocess_pixai(img) for img in batch]).astype(np.float32)
            logits = session.run(None, {input_name: arr})[0]
            probs = _sigmoid(np.asarray(logits))
            for row in probs:
                _collect_pixai(row, tags_arr, is_character, ip_map, general_threshold, character_threshold, include_ip, out)
                pbar.update(1)

    return out


def _collect_pixai(row, tags_arr, is_character, ip_map, general_threshold, character_threshold, include_ip, out):
    char_mask = is_character & (row >= character_threshold)
    gen_mask = (~is_character) & (row >= general_threshold)
    character = tags_arr[char_mask].tolist()
    general = tags_arr[gen_mask].tolist()

    ip_tags: List[str] = []
    if include_ip and ip_map:
        seen = set()
        for tag in character:
            for ip in ip_map.get(tag, []):
                if ip not in seen:
                    seen.add(ip)
                    ip_tags.append(ip)

    out.append(character + ip_tags + general)


# ---------- Nodes ----------

class WD14TaggerNode:
    @classmethod
    def INPUT_TYPES(cls):
        installed: List[str] = []
        if os.path.isdir(WD14_DIR):
            for name in os.listdir(WD14_DIR):
                if name.endswith(".onnx"):
                    base = os.path.splitext(name)[0]
                    if os.path.exists(os.path.join(WD14_DIR, base + ".csv")):
                        installed.append(base)
        models = list(dict.fromkeys(WD14_MODELS + installed)) or [DEFAULT_WD14_MODEL]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (models, {"default": DEFAULT_WD14_MODEL}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "separator": ("STRING", {"default": ", ", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "include_rating": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "force_download": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    CATEGORY = "booru-helper-mini"

    def tag(self, image, model, batch_size, threshold, character_threshold, separator,
            exclude_tags, replace_underscore, include_rating, trailing_comma, force_download):
        images = _tensor_to_pil_batch(image)
        exclude = _exclude_set(exclude_tags)
        results = _run_wd14(images, model, batch_size, threshold, character_threshold, include_rating, force_download)
        strings = [_format(tags, exclude, separator, trailing_comma, replace_underscore) for tags in results]
        return (strings,)


class CamieTaggerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.492, "min": 0.0, "max": 1.0, "step": 0.001}),
                "character_threshold": ("FLOAT", {"default": 0.492, "min": 0.0, "max": 1.0, "step": 0.001}),
                "separator": ("STRING", {"default": ", ", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "force_download": ("BOOLEAN", {"default": False}),
                "repo_id": ("STRING", {"default": DEFAULT_CAMIE_REPO, "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    CATEGORY = "booru-helper-mini"

    def tag(self, image, batch_size, threshold, character_threshold, separator,
            exclude_tags, replace_underscore, trailing_comma, force_download, repo_id):
        images = _tensor_to_pil_batch(image)
        exclude = _exclude_set(exclude_tags)
        results = _run_camie(images, repo_id, batch_size, threshold, character_threshold, force_download)
        strings = [_format(tags, exclude, separator, trailing_comma, replace_underscore) for tags in results]
        return (strings,)


class PixAITaggerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "backend": (["pytorch", "onnx"], {"default": "pytorch"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "separator": ("STRING", {"default": ", ", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "include_ip": ("BOOLEAN", {"default": True}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "force_download": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "fp16": ("BOOLEAN", {"default": True}),
                "pytorch_repo_id": ("STRING", {"default": DEFAULT_PIXAI_TORCH_REPO, "multiline": False}),
                "onnx_repo_id": ("STRING", {"default": DEFAULT_PIXAI_ONNX_REPO, "multiline": False}),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    CATEGORY = "booru-helper-mini"

    def tag(self, image, backend, batch_size, threshold, character_threshold, separator,
            exclude_tags, include_ip, replace_underscore, trailing_comma, force_download,
            device, fp16, pytorch_repo_id, onnx_repo_id, hf_token):
        images = _tensor_to_pil_batch(image)
        exclude = _exclude_set(exclude_tags)
        token = _resolve_hf_token(hf_token)
        repo_id = pytorch_repo_id if backend == "pytorch" else onnx_repo_id
        results = _run_pixai(
            images=images,
            backend=backend,
            repo_id=repo_id,
            batch_size=batch_size,
            general_threshold=threshold,
            character_threshold=character_threshold,
            include_ip=include_ip,
            device_choice=device,
            fp16=fp16,
            token=token,
            force_download=force_download,
        )
        strings = [_format(tags, exclude, separator, trailing_comma, replace_underscore) for tags in results]
        return (strings,)


NODE_CLASS_MAPPINGS = {
    "WD14Tagger|adbrasi": WD14TaggerNode,
    "CamieTagger|adbrasi": CamieTaggerNode,
    "PixAITagger|adbrasi": PixAITaggerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14Tagger|adbrasi": "WD14 Tagger",
    "CamieTagger|adbrasi": "Camie Tagger",
    "PixAITagger|adbrasi": "PixAI Tagger",
}
