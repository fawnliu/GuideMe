import os
import json
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.api_config import CONFIG_PATH, resolve_api_credentials


class GeminiAPIGenerator:
    # Credentials and endpoint are resolved via tools.api_config so that no
    # secrets are committed to the repository. They come from (first wins):
    #   1. env vars LLM_API_KEY / LLM_API_HOST
    #   2. api_setting.yaml at the repo root (gitignored; copy from
    #      api_setting.example.yaml and fill in your own key/host)
    DEFAULT_MODEL = "gemini-3-pro-preview"

    OPENAI_COMPAT_PREFIXES = ("doubao", "gpt", "claude", "deepseek", "qwen")

    def __init__(self, model: str = DEFAULT_MODEL):
        self.API_KEY, self.API_HOST = resolve_api_credentials()
        if not self.API_KEY:
            raise RuntimeError(
                "LLM API key is not set. Either export LLM_API_KEY=... or add "
                f"api_key to {CONFIG_PATH} (copy api_setting.example.yaml first)."
            )
        if not self.API_HOST:
            raise RuntimeError(
                "LLM API host is not set. Either export LLM_API_HOST=... or add "
                f"api_host to {CONFIG_PATH} (e.g. api_host: api.example.com)."
            )
        self.model = model
        self.default_sampling_args = dict(n=1, temperature=0.5, top_p=0.95, max_tokens=8192)
        self._token_lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._api_calls = 0
        self._ssl_verify = self._resolve_ssl_verify()

    @classmethod
    def build(cls, model_id: str = DEFAULT_MODEL, **kwargs):
        return cls(model=model_id)

    def _use_openai_format(self) -> bool:
        return any(self.model.startswith(p) for p in self.OPENAI_COMPAT_PREFIXES)

    def _resolve_ssl_verify(self):
        # Debug only: explicitly disable certificate verification.
        if os.getenv("INFER_SSL_NO_VERIFY", "0").lower() in {"1", "true", "yes"}:
            return False

        ca_file = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
        if ca_file:
            return ca_file

        try:
            import certifi

            return certifi.where()
        except Exception:
            return True

    def _call_api(self, system_prompt: str, messages: list[dict]) -> str:
        if self._use_openai_format():
            return self._call_api_openai(system_prompt, messages)
        return self._call_api_gemini(system_prompt, messages)

    # ── OpenAI-compatible format (doubao, gpt, claude, etc.) ──

    def _call_api_openai(self, system_prompt: str, messages: list[dict]) -> str:
        endpoint = "/v1/chat/completions"
        url = f"https://{self.API_HOST}{endpoint}"

        oai_messages: list[dict] = []
        if system_prompt:
            oai_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = "assistant" if msg["role"] == "model" else msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                oai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                oai_parts: list[dict] = []
                for part in content:
                    if "text" in part:
                        oai_parts.append({"type": "text", "text": part["text"]})
                    elif "inlineData" in part:
                        mime = part["inlineData"]["mimeType"]
                        data = part["inlineData"]["data"]
                        oai_parts.append({
                            "type": "video_url",
                            "video_url": {"url": f"data:{mime};base64,{data}"},
                        })
                oai_messages.append({"role": role, "content": oai_parts})

        payload = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": self.default_sampling_args["temperature"],
            "max_tokens": self.default_sampling_args["max_tokens"],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response = requests.post(
            url,
            data=payload_bytes,
            headers=headers,
            timeout=1200,
            verify=self._ssl_verify,
        )
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text[:500]}")

        result = response.json()
        text = result["choices"][0]["message"]["content"]

        usage = result.get("usage", {})
        with self._token_lock:
            self._input_tokens += usage.get("prompt_tokens", 0)
            self._output_tokens += usage.get("completion_tokens", 0)
            self._total_tokens += usage.get("total_tokens", 0)
            self._api_calls += 1

        return text

    # ── Gemini native format ──

    def _call_api_gemini(self, system_prompt: str, messages: list[dict]) -> str:
        endpoint = f"/v1beta/models/{self.model}:generateContent?key={self.API_KEY}"
        url = f"https://{self.API_HOST}{endpoint}"

        contents = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            content = msg["content"]
            if isinstance(content, list):
                parts = content
            else:
                parts = [{"text": content}]
            contents.append({"role": role, "parts": parts})

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
        }

        headers = {"Content-Type": "application/json", "User-Agent": "VideoAnalyzer/1.0"}
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response = requests.post(
            url,
            data=payload_bytes,
            headers=headers,
            timeout=1200,
            verify=self._ssl_verify,
        )
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text[:500]}")

        result = response.json()
        parts = result["candidates"][0]["content"]["parts"]

        usage = result.get("usageMetadata", {})
        with self._token_lock:
            self._input_tokens += usage.get("promptTokenCount", 0)
            self._output_tokens += (
                usage.get("candidatesTokenCount", 0)
                + usage.get("thoughtsTokenCount", 0)
            )
            self._total_tokens += usage.get("totalTokenCount", 0)
            self._api_calls += 1

        texts = [
            p.get("text", "")
            for p in parts
            if not p.get("thought", False) and p.get("text", "")
        ]
        return "".join(texts) if texts else parts[-1].get("text", "")

    def get_token_stats(self) -> dict:
        with self._token_lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._total_tokens,
                "api_calls": self._api_calls,
            }

    def _parse_inputs(self, inputs: list[tuple[str, str]]) -> tuple[str, list[dict]]:
        system_prompt = ""
        messages = []
        for role, content in inputs:
            if role == "system":
                system_prompt = content
            else:
                messages.append({"role": role, "content": content})
        return system_prompt, messages

    def generate(self, inputs: list[tuple[str, str]], **kwargs) -> list[str]:
        n = kwargs.get("n", self.default_sampling_args.get("n", 1))
        if n <= 0:
            return []
        system_prompt, messages = self._parse_inputs(inputs)

        if n == 1:
            return [self._call_api(system_prompt, messages)]

        # Make n parallel API calls for multiple samples
        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [executor.submit(self._call_api, system_prompt, messages) for _ in range(n)]
            return [f.result() for f in futures]

    def batch_generate(
        self, inputs: list[list[tuple[str, str]]], **kwargs
    ) -> list[list[str]]:
        if not inputs:
            return []
        parsed = [self._parse_inputs(inp) for inp in inputs]

        # Run all batch items in parallel, each returning a single response
        with ThreadPoolExecutor(max_workers=min(len(inputs), 8)) as executor:
            futures = {
                executor.submit(self._call_api, sys_prompt, msgs): i
                for i, (sys_prompt, msgs) in enumerate(parsed)
            }
            results = [None] * len(inputs)
            for future in as_completed(futures):
                i = futures[future]
                results[i] = [future.result()]

        return results


class LLMGenerator:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.default_sampling_args = dict(
            n=1, temperature=0.5, top_p=0.95, max_tokens=128 * 1024  # 128K
        )
        global SamplingParams
        from vllm import SamplingParams

    @classmethod
    def build(cls, model_id: str, number_gpus: int, local_rank: int | None = None):
        if local_rank is not None:
            # trick to use vLLM in slurm environment
            gpu_ids = [local_rank * number_gpus + i for i in range(number_gpus)]
            gpus = ",".join(str(g) for g in gpu_ids)
            # print process id
            print(f"Process ID: {os.getpid()}")
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM
        from transformers import AutoTokenizer

        llm = LLM(
            model=model_id,
            tensor_parallel_size=number_gpus,
            gpu_memory_utilization=0.96,
            swap_space=0,
            max_num_seqs=64,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return cls(llm, tokenizer)

    def generate(
        self, inputs: list[tuple[str, str]], **vllm_sampling_args
    ) -> list[str]:

        messages = [{"role": r, "content": c} for r, c in inputs]
        prompts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        sampling_args = self.default_sampling_args.copy()
        sampling_args.update(vllm_sampling_args)
        sampling_params = SamplingParams(**sampling_args)
        num_repeats = sampling_args.get("n", 1)

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        texts = [outputs[0].outputs[i].text for i in range(num_repeats)]

        return texts

    def batch_generate(
        self, inputs: list[list[tuple[str, str]]], **vllm_sampling_args
    ) -> list[list[str]]:
        messages = [[{"role": r, "content": c} for r, c in batch] for batch in inputs]
        prompts = [
            self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            for m in messages
        ]

        sampling_args = self.default_sampling_args.copy()
        sampling_args.update(vllm_sampling_args)
        sampling_params = SamplingParams(**sampling_args)
        num_repeats = sampling_args.get("n", 1)

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        texts = [
            [outputs[j].outputs[i].text for i in range(num_repeats)]
            for j in range(len(inputs))
        ]

        return texts
