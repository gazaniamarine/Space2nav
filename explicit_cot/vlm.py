"""Standalone explicit-CoT VLM wrapper.

Generalizes the load/generate pattern from LemNAV's runtime/supervisor.py
(_ensure_loaded/_ask_json) and run_rover_multileg_cot.py's
CoTTaskSupervisor._generate_raw into one reusable class that isn't tied to
any TaskSupervisor subclassing/monkeypatch machinery. Callers format their
own prompt (asking for <reasoning>...</reasoning><goal>...</goal>) and get
back the parsed (reasoning_text, goal_value) pair.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .parsing import extract_json_value, split_reasoning

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 768


class ExplicitCoTVLM:
    """Deterministic (do_sample=False) chat-template VLM call, returning
    explicit chain-of-thought text split from its tagged JSON payload."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda:0",
                 load_in_4bit: bool = False, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import importlib.util

        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _VLM
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLM

        use_4bit = self.load_in_4bit
        if use_4bit and importlib.util.find_spec("bitsandbytes") is None:
            print("[explicit_cot] bitsandbytes not available -- falling back to fp16 load")
            use_4bit = False
        kwargs = {"torch_dtype": torch.float16}
        if use_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        print(f"[explicit_cot] loading {self.model_id} (4bit={use_4bit}) ...")
        self._model = _VLM.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def ask(self, prompt: str, images: Optional[List[np.ndarray]] = None,
            max_new_tokens: Optional[int] = None,
            tag: str = "vlm") -> "tuple[str, Optional[Union[dict, list]]]":
        """Run one deterministic generation, log+return (reasoning_text, parsed_goal).

        parsed_goal is None if the <goal> JSON failed to parse even after a
        best-effort fallback over the full reply.
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        pil = [Image.fromarray(np.asarray(im).astype("uint8")) for im in (images or [])]
        content = [{"type": "image", "image": im} for im in pil]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        kw = dict(text=[text], return_tensors="pt")
        if pil:
            kw["images"] = pil
        inputs = self._processor(**kw).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False)
        reply = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        reasoning, goal_text = split_reasoning(reply)
        print(f"[cot-reason:{tag}] {reasoning or '(no <reasoning> block found in reply)'}")
        value = extract_json_value(goal_text)
        if value is None:
            value = extract_json_value(reply)  # tags missing/malformed -- best effort
        return reasoning, value
