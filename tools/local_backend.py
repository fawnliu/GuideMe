"""
Local HuggingFace backend, a drop-in replacement for GeminiAPIGenerator.

Only exposes the two interfaces infer.py actually uses:
    - _call_api(system_prompt, messages) -> str
    - get_token_stats() -> dict

So infer.py's main inference flow (iterate files -> process_sample -> merge -> save)
needs no changes; only the "backend model" is swapped from the cloud Gemini API to a local Qwen3-VL.

messages follow the format produced by build_gemini_messages in infer.py:
    [{"role": "user"/"assistant", "content": str | list[part]}]
where a part can be:
    - {"text": "..."}
    - {"videoPath": "/abs/path.mp4", "fps": 2.0}      # local backend prefers a path
    - {"inlineData": {"mimeType": "video/mp4", "data": <base64>}}  # API-format compatible
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional


class LocalQwenVLGenerator:
    """Local Qwen3-VL inference engine (HF transformers)."""

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bf16",
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        fps: float = 2.0,
        max_pixels: Optional[int] = 384 * 28 * 28,
    ):
        import torch  # deferred import; API mode does not need torch

        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.fps = float(fps)
        self.max_pixels = max_pixels
        self._torch = torch

        dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(str(torch_dtype).lower(), torch.bfloat16)

        from transformers import AutoProcessor

        try:
            from transformers import (
                Qwen3VLForConditionalGeneration,
                Qwen3VLMoeForConditionalGeneration,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "The current transformers does not support Qwen3-VL; install transformers>=4.57 "
                "(this repo's pyproject.toml pins transformers==4.57.3; just run `uv sync`)."
            ) from exc

        print(f"[local backend] loading Qwen3-VL from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_cls = (
            Qwen3VLMoeForConditionalGeneration
            if "A3B" in model_path
            else Qwen3VLForConditionalGeneration
        )
        self.model = model_cls.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.model.eval()

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Qwen3-VL requires qwen_vl_utils; run `pip install qwen-vl-utils` first."
            ) from exc
        self.process_vision_info = process_vision_info

        # HF model.generate is not thread-safe; serialize with a lock (even if the caller uses multiple workers)
        self._gen_lock = threading.Lock()

        # token statistics
        self._stat_lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._api_calls = 0

        # base64 -> temp-file cache (only used when the caller passes inlineData)
        self._tmp_lock = threading.Lock()
        self._tmp_cache: Dict[str, str] = {}

        print("[local backend] model loaded and ready.")

    # ------------------------------------------------------------------ #
    # Message format conversion: Gemini-style -> Qwen3-VL chat-template messages
    # ------------------------------------------------------------------ #
    def _b64_to_tempfile(self, data_b64: str) -> str:
        key = hashlib.md5(data_b64.encode("utf-8")).hexdigest()
        with self._tmp_lock:
            cached = self._tmp_cache.get(key)
            if cached and os.path.exists(cached):
                return cached
        fd, path = tempfile.mkstemp(suffix=".mp4", prefix="local_infer_")
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(data_b64))
        with self._tmp_lock:
            self._tmp_cache[key] = path
        return path

    def _video_part(self, video_path: str, fps: Optional[float] = None) -> Dict[str, Any]:
        part: Dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "fps": float(fps if fps is not None else self.fps),
        }
        if self.max_pixels is not None:
            part["max_pixels"] = int(self.max_pixels)
        return part

    def _to_qwen_messages(
        self, system_prompt: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        qwen_messages: List[Dict[str, Any]] = []
        if system_prompt:
            qwen_messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        for msg in messages:
            role = msg.get("role", "user")
            if role == "model":
                role = "assistant"
            content = msg.get("content", "")

            parts: List[Dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part and part["text"]:
                        parts.append({"type": "text", "text": part["text"]})
                    elif "videoPath" in part:
                        parts.append(self._video_part(part["videoPath"], part.get("fps")))
                    elif "inlineData" in part:
                        path = self._b64_to_tempfile(part["inlineData"]["data"])
                        parts.append(self._video_part(path, part.get("fps")))

            if parts:
                qwen_messages.append({"role": role, "content": parts})

        return qwen_messages

    # ------------------------------------------------------------------ #
    # Interfaces aligned with GeminiAPIGenerator
    # ------------------------------------------------------------------ #
    def _call_api(self, system_prompt: str, messages: List[Dict[str, Any]]) -> str:
        qwen_messages = self._to_qwen_messages(system_prompt, messages)

        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        # Qwen3-VL builds the prompt with frame timestamps and needs each video's video_metadata (with the true fps).
        # return_video_metadata=True makes process_vision_info return videos as
        # [(video_tensor, metadata_dict), ...] and sets do_sample_frames=False in video_kwargs.
        images, videos, video_kwargs = self.process_vision_info(
            qwen_messages, return_video_kwargs=True, return_video_metadata=True
        )

        proc_kwargs: Dict[str, Any] = {
            "text": [text],
            "images": images,
            "return_tensors": "pt",
        }

        if videos is not None:
            video_tensors: List[Any] = []
            video_metadata: List[Any] = []
            for item in videos:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    video_tensors.append(item[0])
                    video_metadata.append(item[1])
                else:
                    video_tensors.append(item)
            proc_kwargs["videos"] = video_tensors
            videos_kwargs = dict(video_kwargs) if video_kwargs else {}
            if video_metadata:
                videos_kwargs["video_metadata"] = video_metadata
            proc_kwargs["videos_kwargs"] = videos_kwargs

        inputs = self.processor(**proc_kwargs).to(self.model.device)

        do_sample = self.temperature > 0
        with self._gen_lock, self._torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=(max(self.temperature, 1e-6) if do_sample else None),
            )

        input_len = int(inputs.input_ids.shape[1])
        trimmed = generated_ids[:, input_len:]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        output_len = int(trimmed.shape[1])
        with self._stat_lock:
            self._input_tokens += input_len
            self._output_tokens += output_len
            self._total_tokens += input_len + output_len
            self._api_calls += 1

        return out_text

    def get_token_stats(self) -> Dict[str, int]:
        with self._stat_lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._total_tokens,
                "api_calls": self._api_calls,
            }
