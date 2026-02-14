import csv
import gc
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import comfy.utils
import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths

try:
    import timm
    import torch
    import torchvision.transforms as transforms
except Exception:
    timm = None
    torch = None
    transforms = None

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
DEFAULT_PIXAI_REPO = "pixai-labs/pixai-tagger-v0.9"

WD14_ONNX_NAME = "model.onnx"
WD14_CSV_NAME = "selected_tags.csv"
CAMIE_ONNX_FILE = "camie-tagger-v2.onnx"
CAMIE_META_FILE = "camie-tagger-v2-metadata.json"
PIXAI_PTH_FILE = "model_v0.9.pth"
PIXAI_TAGS_JSON_FILE = "tags_v0.9_13k.json"
PIXAI_CHAR_IP_MAP_FILE = "char_ip_map.json"

if "wd14_tagger" in folder_paths.folder_names_and_paths:
    BASE_MODELS_DIR = folder_paths.get_folder_paths("wd14_tagger")[0]
else:
    BASE_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(BASE_MODELS_DIR, exist_ok=True)

WD14_DIR = os.path.join(BASE_MODELS_DIR, "wd14")
CAMIE_DIR = os.path.join(BASE_MODELS_DIR, "camie")
PIXAI_DIR = os.path.join(BASE_MODELS_DIR, "pixai")
os.makedirs(WD14_DIR, exist_ok=True)
os.makedirs(CAMIE_DIR, exist_ok=True)
os.makedirs(PIXAI_DIR, exist_ok=True)

WD14_SESSION_CACHE: Dict[str, Tuple[ort.InferenceSession, str, str, int]] = {}
WD14_TAG_CACHE: Dict[str, Tuple[List[str], List[int], List[int], List[int]]] = {}
CAMIE_CACHE: Dict[str, Tuple[ort.InferenceSession, str, Dict[int, str], Dict[str, str], int]] = {}
PIXAI_CACHE: Dict[str, Tuple[object, Dict[int, str], int, int, Dict[str, List[str]], str]] = {}


def _cleanup_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _providers() -> List[str]:
    available = ort.get_available_providers()
    ordered: List[str] = []
    if "CUDAExecutionProvider" in available:
        ordered.append("CUDAExecutionProvider")
    if "ROCMExecutionProvider" in available:
        ordered.append("ROCMExecutionProvider")
    if "CPUExecutionProvider" not in ordered:
        ordered.append("CPUExecutionProvider")
    return ordered


def _iterate_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _tensor_to_pil_batch(image_tensor) -> List[Image.Image]:
    arr = image_tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _normalize_models(models_text: str) -> List[str]:
    aliases = {"wd": "wd14", "wd14": "wd14", "camie": "camie", "pixai": "pixai"}
    out: List[str] = []
    for token in models_text.split(","):
        model = aliases.get(token.strip().lower())
        if model and model not in out:
            out.append(model)
    return out or ["wd14"]


def _exclude_set(exclude_tags: str) -> set:
    return {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}


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
    downloaded_onnx = hf_hub_download(repo_id=repo_id, filename=WD14_ONNX_NAME, local_dir=cache_dir, force_download=force_download)
    downloaded_csv = hf_hub_download(repo_id=repo_id, filename=WD14_CSV_NAME, local_dir=cache_dir, force_download=force_download)

    with open(downloaded_onnx, "rb") as src, open(onnx_path, "wb") as dst:
        dst.write(src.read())
    with open(downloaded_csv, "rb") as src, open(csv_path, "wb") as dst:
        dst.write(src.read())

    return onnx_path, csv_path


def _load_wd14_tags(csv_path: str, replace_underscore: bool):
    key = f"{csv_path}:{replace_underscore}"
    if key in WD14_TAG_CACHE:
        return WD14_TAG_CACHE[key]

    tags: List[str] = []
    rating_idx: List[int] = []
    general_idx: List[int] = []
    character_idx: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            tag_name = row[1].replace("_", " ") if replace_underscore else row[1]
            category = row[2]
            tags.append(tag_name)
            if category == "9":
                rating_idx.append(i)
            elif category == "0":
                general_idx.append(i)
            elif category == "4":
                character_idx.append(i)

    WD14_TAG_CACHE[key] = (tags, rating_idx, general_idx, character_idx)
    return WD14_TAG_CACHE[key]


def _get_wd14_session(model_name: str, onnx_path: str):
    key = f"{model_name}:{onnx_path}"
    if key in WD14_SESSION_CACHE:
        return WD14_SESSION_CACHE[key]

    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    try:
        input_size = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    except Exception:
        input_size = 448
    del model

    session = ort.InferenceSession(onnx_path, providers=_providers())
    output_name = session.get_outputs()[0].name
    WD14_SESSION_CACHE[key] = (session, input_name, output_name, input_size)
    return WD14_SESSION_CACHE[key]


def _preprocess_wd14(img: Image.Image, size: int) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    ratio = float(size) / max(img.size)
    new_size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))

    arr = np.array(square, dtype=np.float32)
    return arr[:, :, ::-1]


def _run_wd14(images: List[Image.Image], model_name: str, batch_size: int, threshold: float, character_threshold: float, replace_underscore: bool, include_rating: bool, force_download: bool) -> List[List[str]]:
    onnx_path, csv_path = _ensure_wd14_assets(model_name, force_download)
    if force_download:
        WD14_SESSION_CACHE.pop(f"{model_name}:{onnx_path}", None)
    tags, rating_idx, general_idx, character_idx = _load_wd14_tags(csv_path, replace_underscore)
    session, input_name, output_name, input_size = _get_wd14_session(model_name, onnx_path)

    out: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))
    for batch in _iterate_batches(images, batch_size):
        arr = np.stack([_preprocess_wd14(img, input_size) for img in batch]).astype(np.float32)
        probs = session.run([output_name], {input_name: arr})[0]
        for row in probs:
            general = [tags[i] for i in general_idx if row[i] >= threshold]
            character = [tags[i] for i in character_idx if row[i] >= character_threshold]
            sample = character + general
            if include_rating and rating_idx:
                best = max(rating_idx, key=lambda idx: row[idx])
                sample.insert(0, tags[best])
            out.append(sample)
            pbar.update(1)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_camie_assets(repo_id: str, force_download: bool) -> Tuple[str, str]:
    model_dir = os.path.join(CAMIE_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, CAMIE_ONNX_FILE)
    meta_path = os.path.join(model_dir, CAMIE_META_FILE)

    if force_download or not (os.path.exists(onnx_path) and os.path.exists(meta_path)):
        hf_hub_download(repo_id=repo_id, filename=CAMIE_ONNX_FILE, local_dir=model_dir, force_download=force_download)
        hf_hub_download(repo_id=repo_id, filename=CAMIE_META_FILE, local_dir=model_dir, force_download=force_download)

    return onnx_path, meta_path


def _load_camie_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    mapping = metadata["dataset_info"]["tag_mapping"]
    idx_to_tag = {int(k): v for k, v in mapping["idx_to_tag"].items()}
    tag_to_category = mapping["tag_to_category"]
    img_size = int(metadata.get("model_info", {}).get("img_size", 448))
    return idx_to_tag, tag_to_category, img_size


def _preprocess_imagenet(img: Image.Image, img_size: int) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    w, h = img.size
    ratio = w / max(1, h)
    if ratio > 1:
        nw = img_size
        nh = max(1, int(nw / ratio))
    else:
        nh = img_size
        nw = max(1, int(nh * ratio))

    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (img_size, img_size), (124, 116, 104))
    padded.paste(resized, ((img_size - nw) // 2, (img_size - nh) // 2))

    arr = np.array(padded).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)


def _get_camie(repo_id: str, force_download: bool):
    if repo_id in CAMIE_CACHE and not force_download:
        return CAMIE_CACHE[repo_id]

    onnx_path, meta_path = _ensure_camie_assets(repo_id, force_download)
    idx_to_tag, tag_to_category, img_size = _load_camie_meta(meta_path)
    session = ort.InferenceSession(onnx_path, providers=_providers())
    input_name = session.get_inputs()[0].name

    CAMIE_CACHE[repo_id] = (session, input_name, idx_to_tag, tag_to_category, img_size)
    return CAMIE_CACHE[repo_id]


def _run_camie(images: List[Image.Image], repo_id: str, batch_size: int, general_threshold: float, character_threshold: float, min_confidence: float, force_download: bool) -> List[List[str]]:
    session, input_name, idx_to_tag, tag_to_category, img_size = _get_camie(repo_id, force_download)
    out: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))

    for batch in _iterate_batches(images, batch_size):
        arr = np.stack([_preprocess_imagenet(img, img_size) for img in batch]).astype(np.float32)
        raw = session.run(None, {input_name: arr})
        logits = raw[1] if len(raw) >= 2 else raw[0]
        probs = _sigmoid(logits)

        for row in probs:
            general: List[str] = []
            character: List[str] = []
            for idx, score in enumerate(row):
                if score < min_confidence:
                    continue
                tag = idx_to_tag.get(idx)
                if tag is None:
                    continue
                category = tag_to_category.get(tag, "general").lower()
                threshold = character_threshold if category == "character" else general_threshold
                if score < threshold:
                    continue
                if category == "character":
                    character.append(tag)
                elif category != "rating":
                    general.append(tag)
            out.append(character + general)
            pbar.update(1)

    return out


class PixAITaggingHead(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int):
        if torch is None:
            return
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(input_dim, num_classes))

    def forward(self, x):
        logits = self.head(x)
        return torch.sigmoid(logits)


def _require_pixai_deps() -> None:
    if torch is None or timm is None or transforms is None:
        raise RuntimeError("PixAI requires torch, torchvision and timm")


def _resolve_token(token: str) -> Optional[str]:
    token = token.strip() if token else ""
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

    env = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if env:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = env
    return env


def _ensure_pixai_assets(repo_id: str, hf_token: Optional[str], force_download: bool):
    model_dir = os.path.join(PIXAI_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    for name in [PIXAI_PTH_FILE, PIXAI_TAGS_JSON_FILE, PIXAI_CHAR_IP_MAP_FILE]:
        path = os.path.join(model_dir, name)
        if force_download or not os.path.exists(path):
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=model_dir, force_download=force_download, token=hf_token)

    return (
        os.path.join(model_dir, PIXAI_PTH_FILE),
        os.path.join(model_dir, PIXAI_TAGS_JSON_FILE),
        os.path.join(model_dir, PIXAI_CHAR_IP_MAP_FILE),
    )


def _pixai_to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        img.load()
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "P":
        return _pixai_to_rgb(img.convert("RGBA"))
    return img.convert("RGB")


def _pixai_transform():
    _require_pixai_deps()
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _load_pixai_maps(tags_path: str, ip_map_path: str):
    with open(tags_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tag_map = payload["tag_map"]
    split = payload["tag_split"]
    idx_to_tag = {int(v): k for k, v in tag_map.items()}
    gen_count = int(split["gen_tag_count"])
    char_count = int(split["character_tag_count"])
    with open(ip_map_path, "r", encoding="utf-8") as f:
        ip_map = json.load(f)
    return idx_to_tag, gen_count, char_count, ip_map


def _pixai_device(device: str) -> str:
    _require_pixai_deps()
    if device == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).to("cuda")
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"
    return device


def _build_pixai(weights_path: str, num_classes: int, device: str):
    _require_pixai_deps()
    encoder = timm.create_model("hf_hub:SmilingWolf/wd-eva02-large-tagger-v3", pretrained=False)
    encoder.reset_classifier(0)
    decoder = PixAITaggingHead(1024, num_classes)
    model = torch.nn.Sequential(encoder, decoder)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _get_pixai(repo_id: str, hf_token: Optional[str], force_download: bool, device: str):
    key = f"{repo_id}:{device}"
    if key in PIXAI_CACHE and not force_download:
        return PIXAI_CACHE[key]

    weights, tags, ip_map = _ensure_pixai_assets(repo_id, hf_token, force_download)
    idx_to_tag, gen_count, char_count, char_ip_map = _load_pixai_maps(tags, ip_map)
    model = _build_pixai(weights, len(idx_to_tag), device)
    PIXAI_CACHE[key] = (model, idx_to_tag, gen_count, char_count, char_ip_map, device)
    return PIXAI_CACHE[key]


def _run_pixai(images: List[Image.Image], repo_id: str, batch_size: int, general_threshold: float, character_threshold: float, pixai_no_ip: bool, pixai_device: str, hf_token: Optional[str], force_download: bool) -> List[List[str]]:
    _require_pixai_deps()
    device = _pixai_device(pixai_device)
    model, idx_to_tag, gen_count, char_count, char_ip_map, device = _get_pixai(repo_id, hf_token, force_download, device)
    transform = _pixai_transform()

    out: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))

    for batch in _iterate_batches(images, batch_size):
        tensors = [transform(_pixai_to_rgb(img)) for img in batch]
        batch_tensor = torch.stack(tensors)
        if device == "cuda":
            batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)
        else:
            batch_tensor = batch_tensor.to(device)

        with torch.inference_mode():
            probs = model(batch_tensor)

        for row in probs:
            gen_idx = (row[:gen_count] > general_threshold).nonzero(as_tuple=True)[0]
            char_idx = (row[gen_count : gen_count + char_count] > character_threshold).nonzero(as_tuple=True)[0]

            general = [idx_to_tag[int(i)] for i in gen_idx.cpu().tolist() if int(i) in idx_to_tag]
            character = [idx_to_tag[int(i + gen_count)] for i in char_idx.cpu().tolist() if int(i + gen_count) in idx_to_tag]

            ip_tags: List[str] = []
            if not pixai_no_ip:
                for tag in character:
                    if tag in char_ip_map:
                        ip_tags.extend(char_ip_map[tag])
                ip_tags = sorted(set(ip_tags))

            out.append(character + ip_tags + general)
            pbar.update(1)

    return out


class BooruTagger:
    @classmethod
    def INPUT_TYPES(cls):
        installed = [
            os.path.splitext(name)[0]
            for name in os.listdir(WD14_DIR)
            if name.endswith(".onnx") and os.path.exists(os.path.join(WD14_DIR, os.path.splitext(name)[0] + ".csv"))
        ]
        models = list(dict.fromkeys(WD14_MODELS + installed))
        if not models:
            models = [DEFAULT_WD14_MODEL]

        return {
            "required": {
                "image": ("IMAGE",),
                "selected_models": ("STRING", {"default": "wd14,camie,pixai", "multiline": False}),
                "wd14_model": (models, {"default": DEFAULT_WD14_MODEL}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
                "wd14_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wd14_character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "camie_threshold": ("FLOAT", {"default": 0.492, "min": 0.0, "max": 1.0, "step": 0.01}),
                "camie_character_threshold": ("FLOAT", {"default": 0.492, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pixai_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pixai_character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wd14_separator": ("STRING", {"default": ", ", "multiline": False}),
                "camie_separator": ("STRING", {"default": ", ", "multiline": False}),
                "pixai_separator": ("STRING", {"default": ", ", "multiline": False}),
                "model_separator": ("STRING", {"default": " | ", "multiline": False}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "dedupe": ("BOOLEAN", {"default": True}),
                "include_rating": ("BOOLEAN", {"default": False}),
                "pixai_no_ip": ("BOOLEAN", {"default": False}),
                "force_download": ("BOOLEAN", {"default": False}),
                "pixai_device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "camie_repo_id": ("STRING", {"default": DEFAULT_CAMIE_REPO, "multiline": False}),
                "pixai_repo_id": ("STRING", {"default": DEFAULT_PIXAI_REPO, "multiline": False}),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(
        self,
        image,
        selected_models,
        wd14_model,
        batch_size,
        wd14_threshold,
        wd14_character_threshold,
        camie_threshold,
        camie_character_threshold,
        pixai_threshold,
        pixai_character_threshold,
        wd14_separator,
        camie_separator,
        pixai_separator,
        model_separator,
        exclude_tags,
        replace_underscore,
        dedupe,
        include_rating,
        pixai_no_ip,
        force_download,
        pixai_device,
        camie_repo_id,
        pixai_repo_id,
        hf_token,
    ):
        selected = _normalize_models(selected_models)
        images = _tensor_to_pil_batch(image)
        exclude = _exclude_set(exclude_tags)
        token = _resolve_token(hf_token)

        by_model: Dict[str, List[List[str]]] = {}

        for model in selected:
            if model == "wd14":
                by_model[model] = _run_wd14(
                    images=images,
                    model_name=wd14_model,
                    batch_size=batch_size,
                    threshold=wd14_threshold,
                    character_threshold=wd14_character_threshold,
                    replace_underscore=replace_underscore,
                    include_rating=include_rating,
                    force_download=force_download,
                )
            elif model == "camie":
                by_model[model] = _run_camie(
                    images=images,
                    repo_id=camie_repo_id,
                    batch_size=batch_size,
                    general_threshold=camie_threshold,
                    character_threshold=camie_character_threshold,
                    min_confidence=0.1,
                    force_download=force_download,
                )
            elif model == "pixai":
                by_model[model] = _run_pixai(
                    images=images,
                    repo_id=pixai_repo_id,
                    batch_size=batch_size,
                    general_threshold=pixai_threshold,
                    character_threshold=pixai_character_threshold,
                    pixai_no_ip=pixai_no_ip,
                    pixai_device=pixai_device,
                    hf_token=token,
                    force_download=force_download,
                )
            else:
                raise ValueError(f"Unknown model: {model}")

        sep_by_model = {
            "wd14": wd14_separator,
            "camie": camie_separator,
            "pixai": pixai_separator,
        }

        merged_strings: List[str] = []
        for idx in range(len(images)):
            blocks: List[str] = []
            seen = set()
            for model in selected:
                tags = by_model.get(model, [[]])[idx]
                if exclude:
                    tags = [tag for tag in tags if tag.lower() not in exclude]
                if dedupe:
                    tags = [tag for tag in tags if tag not in seen]
                    seen.update(tags)

                joined = sep_by_model[model].join(tags)
                if joined:
                    blocks.append(joined)

            merged_strings.append(model_separator.join(blocks))

        _cleanup_memory()
        return {"ui": {"tags": merged_strings}, "result": (merged_strings, image)}


NODE_CLASS_MAPPINGS = {
    "BooruTagger|adbrasi": BooruTagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooruTagger|adbrasi": "booru tagger",
}
