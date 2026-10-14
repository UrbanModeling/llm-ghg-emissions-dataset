"""Runs Steps 2-3 (extract.py, verify.py) unchanged with another model. The dataset was built with
--backend deepseek (DeepSeek V4.1 Flash); the other backends ran the model comparison (Table 6 of the paper).

--backend gemini: the Gemini API (e.g. Gemma 4), same client and config as the pipeline, plus an output cap
  (MAX_OUT) so that a model stuck repeating itself fails instead of hanging; a markdown fence around the
  JSON answer (Gemma) is removed, here and for Ollama.
--backend ollama: Ollama Cloud. Same system prompt, temperature 0, thinking off (as for Flash Lite).
  Ollama Cloud ignores the `format` schema for these models, so the response schema (whose field
  descriptions Gemini also sees) is converted to JSON Schema and appended to the system prompt, and JSON
  output is requested.

--backend openrouter: OpenRouter, with the response schema passed as JSON Schema (only providers that
  support it), temperature 0 (no reasoning parameter: providers of non-reasoning models reject it). The cost OpenRouter reports is added up in usage.json.

--backend deepseek: DeepSeek's own API, thinking off, JSON mode; the schema goes in the system prompt as
  for Ollama (the API has no schema-constrained output).
--backend zhipu: Z.ai (GLM) API, as for deepseek, but thinking cannot be turned off: reasoning effort low.
--backend mimo: Xiaomi MiMo API, as for deepseek (thinking off).

Failed papers are retried (up to --passes runs of each step). Time, calls and tokens per step are written
to output/<run>/usage.json. API keys are read from the environment or from a .env file (python-dotenv).

    python eval/run_model.py --backend deepseek --model deepseek-flash --run full --workers 30
    python eval/run_model.py --backend ollama --model gemma4:31b --run b_gm31 --uids-file eval/gold/uids.txt
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "pipeline"))
import extract  # noqa: E402
import verify   # noqa: E402

GenaiClient = extract.genai.Client     # kept before the module's Client is replaced below

OLLAMA_URL = "https://ollama.com/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
ZHIPU_URL = "https://api.z.ai/api/paas/v4/chat/completions"
MIMO_URL = "https://api.xiaomimimo.com/v1/chat/completions"
MAX_OUT = 8192


def json_schema(s) -> dict:
    """google.genai types.Schema -> JSON Schema."""
    d = {"type": s.type.value.lower()}
    if s.description:
        d["description"] = s.description
    if s.enum:
        d["enum"] = list(s.enum)
    if s.properties:
        order = s.property_ordering or list(s.properties)
        d["properties"] = {k: json_schema(s.properties[k]) for k in order}
    if s.required:
        d["required"] = list(s.required)
    if s.items:
        d["items"] = json_schema(s.items)
    return d


def unfence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


class Ollama:
    def __init__(self, model):
        self.key, self.model = os.environ["OLLAMA_API_KEY"], model

    def generate_content(self, model, contents, config):
        schema = json.dumps(json_schema(config.response_schema), ensure_ascii=False)
        system = (f"{config.system_instruction}\n\nAnswer with one JSON object, and nothing else, that follows "
                  f"this JSON Schema:\n{schema}")
        body = {"model": self.model, "stream": False, "think": False, "format": "json",
                "options": {"temperature": config.temperature},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": contents}]}
        for _ in range(3):
            r = requests.post(OLLAMA_URL, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=600)
            if r.status_code < 500 and r.status_code != 429:
                break
        r.raise_for_status()
        j = r.json()
        usage = SimpleNamespace(prompt_token_count=j.get("prompt_eval_count", 0),
                                candidates_token_count=j.get("eval_count", 0))
        return SimpleNamespace(text=unfence(j["message"]["content"]), usage_metadata=usage)


class OpenRouter:
    only = None                        # --or-provider: use this provider only

    def __init__(self, model):
        self.key, self.model = os.environ["OPENROUTER_API_KEY"], model

    def generate_content(self, model, contents, config):
        provider = {"require_parameters": True}
        if self.only:
            provider.update(order=[self.only], allow_fallbacks=False)
        body = {"model": self.model, "temperature": config.temperature,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "response", "schema": json_schema(config.response_schema)}},
                "provider": provider,
                "usage": {"include": True},
                "messages": [{"role": "system", "content": config.system_instruction},
                             {"role": "user", "content": contents}]}
        for _ in range(3):
            r = requests.post(OPENROUTER_URL, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=600)
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(5)
        r.raise_for_status()
        j = r.json()
        u = j["usage"]
        usage = SimpleNamespace(prompt_token_count=u["prompt_tokens"], candidates_token_count=u["completion_tokens"],
                                cost=u.get("cost", 0))
        return SimpleNamespace(text=unfence(j["choices"][0]["message"]["content"]), usage_metadata=usage)


class JsonMode:
    """An OpenAI-style API with JSON mode only: the schema goes in the system prompt."""
    url = key_env = ""
    extra: dict = {}

    def __init__(self, model):
        self.key, self.model = os.environ[self.key_env], model

    def generate_content(self, model, contents, config):
        schema = json.dumps(json_schema(config.response_schema), ensure_ascii=False)
        system = (f"{config.system_instruction}\n\nAnswer with one JSON object, and nothing else, that follows "
                  f"this JSON Schema:\n{schema}")
        body = {"model": self.model, "temperature": config.temperature, **self.extra,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": contents}]}
        for _ in range(3):
            r = requests.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=600)
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(5)
        r.raise_for_status()
        j = r.json()
        usage = SimpleNamespace(prompt_token_count=j["usage"]["prompt_tokens"],
                                candidates_token_count=j["usage"]["completion_tokens"])
        return SimpleNamespace(text=unfence(j["choices"][0]["message"]["content"]), usage_metadata=usage)


class DeepSeek(JsonMode):
    url, key_env, extra = DEEPSEEK_URL, "DEEPSEEK_API_KEY", {"thinking": {"type": "disabled"}}


class Mimo(JsonMode):
    url, key_env, extra = MIMO_URL, "MIMO_API_KEY", {"thinking": {"type": "disabled"}}


class Zhipu(JsonMode):
    url, key_env, extra = ZHIPU_URL, "ZHIPU_API_KEY", {"reasoning_effort": "low"}


class Gemini:
    def __init__(self, model):
        self.client, self.model = GenaiClient(api_key=os.environ["GEMINI_API_KEY"]), model

    def generate_content(self, model, contents, config):
        config = config.model_copy(update={"max_output_tokens": MAX_OUT})
        r = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        u = r.usage_metadata
        usage = SimpleNamespace(prompt_token_count=u.prompt_token_count or 0,
                                candidates_token_count=(u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
        return SimpleNamespace(text=unfence(r.text), usage_metadata=usage)


class Counter:
    """Wraps a backend and counts successful calls and tokens."""
    def __init__(self, backend):
        self.backend, self.lock = backend, threading.Lock()
        self.calls = self.tok_in = self.tok_out = 0
        self.cost = 0.0

    def generate_content(self, **kw):
        r = self.backend.generate_content(**kw)
        with self.lock:
            self.calls += 1
            self.tok_in += r.usage_metadata.prompt_token_count or 0
            self.tok_out += r.usage_metadata.candidates_token_count or 0
            self.cost += getattr(r.usage_metadata, "cost", 0) or 0
        return r


def remaining(path: Path, uids) -> int:
    done = {d["uid"] for d in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else set()
    return len(set(uids) - done)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Steps 2-3 with another model")
    p.add_argument("--backend", choices=["gemini", "ollama", "openrouter", "deepseek", "zhipu", "mimo"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--uids-file", help="papers to run (default: all papers in output/prepared.csv)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--steps", default="extract,verify")
    p.add_argument("--or-provider", help="OpenRouter: use only this provider (e.g. novita)")
    a = p.parse_args()
    OpenRouter.only = a.or_provider
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE / ".env")
    except ImportError:
        pass
    os.environ.setdefault("GEMINI_API_KEY", "")
    out = BASE / "output" / a.run
    uids = ([int(x) for x in open(a.uids_file).read().split()] if a.uids_file else
            pd.read_csv(BASE / "output" / "prepared.csv", usecols=["uid"]).uid.tolist())
    usage_file = out / "usage.json"
    usage = json.loads(usage_file.read_text(encoding="utf-8")) if usage_file.exists() else {}
    usage.update({"model": a.model, "backend": a.backend, "workers": a.workers})

    for step in a.steps.split(","):
        counter = Counter({"gemini": Gemini, "ollama": Ollama, "openrouter": OpenRouter,
                           "deepseek": DeepSeek, "zhipu": Zhipu, "mimo": Mimo}[a.backend](a.model))
        client = SimpleNamespace(models=counter)
        for m in (extract, verify):
            m.genai.Client = lambda api_key=None: client
        extract.MODEL = a.model
        t0 = time.perf_counter()
        for _ in range(a.passes):
            if step == "extract":
                extract.main(a.run, a.uids_file, a.workers)
                left = remaining(out / "extracted.json", uids)
            else:
                verify.main(a.run, a.workers, a.model)
                ext = pd.read_csv(out / "extracted.csv", usecols=["uid"])
                left = remaining(out / "verified.json", ext.uid.unique().tolist())
            if not left:
                break
        usage[step] = {"minutes": round((time.perf_counter() - t0) / 60, 2), "calls": counter.calls,
                       "tokens_in": counter.tok_in, "tokens_out": counter.tok_out, "cost_usd": round(counter.cost, 4),
                       "unfinished": left}
        print(step, usage[step], flush=True)
    usage_file.write_text(json.dumps(usage, indent=1), encoding="utf-8")
