"""Qwen-as-detector: open-vocabulary grounding using the SAME Qwen3-VL
model/weights already loaded for planning/execution reasoning (see
explicit_cot.vlm.ExplicitCoTVLM) -- no Grounding-DINO, no second model
instance.

Vendored (Detection/_overlap/_class_agnostic_nms/_parse_detections/_PROMPT
logic unchanged) from
/home/gpu/Desktop/Nihil/LemNAV/nav_pipeline/qwen_grounding_detector.py and
dino_detector.py's Detection dataclass. That module's docstring already
documents the key empirical finding this relies on: "Qwen3-VL's grounding
output is normalized to a 0-1000 grid regardless of the actual input
resolution (verified empirically 2026-10-09 against RefCOCOg GT boxes)".

Difference from the LemNAV original: that module's Qwen3VLGroundingDetector
loads its OWN model/processor (makes sense there, since it's a drop-in
replacement for a separate GroundingDinoDetector instance in a pipeline
that otherwise never loads a VLM at all). Here we already have an
ExplicitCoTVLM loaded for the stagewise planner/executor, so
detect_with_vlm() below takes that instance and reuses its model/processor
directly instead of loading a second copy -- same weights, same VRAM,
just a different (plain, non-tagged) prompt for this one call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class Detection:
    box: np.ndarray  # [x0, y0, x1, y1] pixels in the input image
    score: float
    label: str

    @property
    def center(self):
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _class_agnostic_nms(dets: List[Detection], iou_threshold: float) -> List[Detection]:
    """Greedily drop lower-score boxes that heavily overlap a kept one
    (label-blind -- a high-IoU pair is overwhelmingly one duplicate/
    mislabeled instance, not two distinct co-located objects)."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for d in dets:
        if all(_overlap(d.box, k.box) < iou_threshold for k in kept):
            kept.append(d)
    return kept


_BOX_RE = re.compile(
    r"\{[^{}]*?\"bbox_2d\"\s*:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,"
    r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\][^{}]*?\}",
    re.DOTALL,
)
_LABEL_RE = re.compile(r"\"label\"\s*:\s*\"([^\"]*)\"")
_SCORE_RE = re.compile(r"\"score\"\s*:\s*(-?\d+\.?\d*)")
_PLAIN_BOX_RE = re.compile(
    r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
)

_PROMPT = (
    "Find every instance of any of these objects/phrases in the image:\n{phrases}\n\n"
    "Respond with ONLY a JSON list (no markdown fences, no prose). Each element:\n"
    '{{"label": "<which phrase it matches>", "bbox_2d": [x1, y1, x2, y2], "score": <0.0-1.0 confidence>}}\n'
    "Coordinates are normalized to a 0-1000 grid, (0,0) top-left, (1000,1000) bottom-right. "
    "If nothing matches, respond with []."
)


def _parse_detections(text: str, w: int, h: int, fallback_label: str) -> List[Detection]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text

    out: List[Detection] = []
    try:
        items = json.loads(text)
        if isinstance(items, dict):
            items = [items]
        for it in items:
            box = it.get("bbox_2d") or it.get("bbox")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            out.append(Detection(
                box=np.array([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], dtype=np.float32),
                score=float(it.get("score", 0.5)),
                label=str(it.get("label", fallback_label)),
            ))
        if out:
            return out
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        pass

    # Regex fallback: the model answered with box objects but not valid JSON.
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        chunk = m.group(0)
        lbl_m = _LABEL_RE.search(chunk)
        score_m = _SCORE_RE.search(chunk)
        out.append(Detection(
            box=np.array([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], dtype=np.float32),
            score=float(score_m.group(1)) if score_m else 0.5,
            label=lbl_m.group(1) if lbl_m else fallback_label,
        ))
    if out:
        return out

    # Last resort: a single bare [x1,y1,x2,y2] with no surrounding object.
    m = _PLAIN_BOX_RE.search(text)
    if m:
        x1, y1, x2, y2 = (float(v) for v in m.groups())
        out.append(Detection(
            box=np.array([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h], dtype=np.float32),
            score=0.5, label=fallback_label,
        ))
    return out


def _phrases(text: str) -> List[str]:
    return [p.strip() for p in text.split(".") if p.strip()]


def detect_with_vlm(vlm, image: np.ndarray, text: str,
                     box_threshold: float = 0.35, nms_iou_threshold: float = 0.4,
                     max_new_tokens: int = 256) -> List[Detection]:
    """image: HxWx3 uint8 RGB. `vlm` is an explicit_cot.vlm.ExplicitCoTVLM --
    reuses its already-loaded model/processor with a plain (non-CoT-tagged)
    grounding prompt. Returns detections sorted by score desc."""
    import torch

    vlm._ensure_loaded()
    h, w = image.shape[:2]
    pil = Image.fromarray(image)
    phrases = _phrases(text) or [text.strip()]
    prompt = _PROMPT.format(phrases="\n".join(f"- {p}" for p in phrases))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": prompt},
    ]}]
    inputs = vlm._processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(vlm._model.device)
    with torch.no_grad():
        gen = vlm._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out_text = vlm._processor.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    fallback_label = phrases[0] if phrases else text.strip()
    dets = _parse_detections(out_text, w, h, fallback_label)
    dets = [d for d in dets if d.score >= box_threshold] or dets  # never drop a sole low-score hit
    dets = _class_agnostic_nms(dets, nms_iou_threshold)
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


def detect_best_with_vlm(vlm, image: np.ndarray, text: str, **kwargs) -> Optional[Detection]:
    dets = detect_with_vlm(vlm, image, text, **kwargs)
    return dets[0] if dets else None
