from __future__ import annotations

import time
from pathlib import Path
from typing import Any

PROCESSOR_OVERRIDES = {
    # The SFT repository was saved with the development-only class name
    # "Qwen2_5_VLImageProcessor". Transformers 4.49 expects
    # "Qwen2VLImageProcessor". The checkpoint is fine-tuned from the Qwen
    # model below, so its official processor is the compatible equivalent.
    "Fancy-MLLM/R1-Onevision-7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def processor_id_for_model(model_id: str) -> str:
    return PROCESSOR_OVERRIDES.get(model_id, model_id)


class R1ModelRunner:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        attention: str = "sdpa",
        seed: int = 0,
    ) -> None:
        # Keep heavyweight GPU/runtime dependencies lazy so scoring, data
        # preparation, and unit tests remain importable on CPU-only systems.
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        self._process_vision_info = process_vision_info
        self.model_id = model_id
        self.processor_id = processor_id_for_model(model_id)
        self.device = device
        self.seed = seed
        self.processor = AutoProcessor.from_pretrained(
            self.processor_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention,
        ).to(device)
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def generate(
        self,
        *,
        image_paths: list[str | Path],
        prompt: str,
        generation: dict[str, Any],
        seed: int | None = None,
    ) -> dict[str, Any]:
        image_content = [
            {"type": "image", "image": str(Path(path).resolve())}
            for path in image_paths
        ]
        messages = [
            {
                "role": "user",
                "content": [*image_content, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        active_seed = self.seed if seed is None else int(seed)
        self._torch.manual_seed(active_seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(active_seed)
            self._torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=True,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                use_cache=True,
            )
        elapsed = time.perf_counter() - started

        prompt_length = int(inputs.input_ids.shape[1])
        trimmed = generated_ids[:, prompt_length:]
        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        peak_gb = (
            self._torch.cuda.max_memory_allocated() / (1024**3)
            if self._torch.cuda.is_available()
            else 0.0
        )
        return {
            "output_text": output,
            "prompt_tokens": prompt_length,
            "output_tokens": int(trimmed.shape[1]),
            "latency_seconds": elapsed,
            "peak_memory_gb": peak_gb,
            "model_revision": self.revision,
            "processor_id": self.processor_id,
            "generation_seed": active_seed,
        }
