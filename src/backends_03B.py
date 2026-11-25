import os
import time
import csv
import json
import statistics
import sys

from typing import Protocol
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime, timezone
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class Generation:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    model: str
    mode: str


class LLMBackend(Protocol):
    def generate(
        self, system: str, user: str, max_tokens: int = 160, *,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float = 1.0, seed: int | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> Generation:
        ...


def validate_sampling(
    max_tokens: int, temperature: float | None, top_k: int | None,
    top_p: float, seed: int | None,
) -> None:
    """temperature=None oznacza greedy; liczba większa od zera próbkowanie."""
    if max_tokens < 1:
        raise ValueError("max_tokens musi być dodatnie")
    if not 0 < top_p <= 1:
        raise ValueError("top_p musi należeć do przedziału (0, 1]")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k musi być dodatnie albo None")
    if temperature is None:
        if top_k is not None or top_p != 1.0 or seed is not None:
            raise ValueError("W trybie greedy pomiń top_k i seed oraz użyj top_p=1.0")
    elif temperature <= 0:
        raise ValueError("Temperatura próbkowania musi być większa od zera; None oznacza greedy")


def gemini_output_schema(model_type: type[BaseModel]) -> dict:
    """Podzbiór schematu Pydantic akceptowany przez wybrany interfejs Gemini.

    W tej lekcji usuwamy ograniczenia długości napisów. Aplikacja nadal
    egzekwuje pełny model Pydantic po odebraniu odpowiedzi.
    """
    def without_string_limits(node):
        if isinstance(node, dict):
            return {
                key: without_string_limits(value)
                for key, value in node.items()
                if key not in {"minLength", "maxLength"}
            }
        if isinstance(node, list):
            return [without_string_limits(value) for value in node]
        return node

    return without_string_limits(model_type.model_json_schema())


class ApiBackend:
    def __init__(self) -> None:
        from google import genai
        # OpenAI: from openai import OpenAI

        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.supports_seed = True
        # OpenAI: self.model = os.environ["OPENAI_MODEL"]
        # OpenAI: self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # OpenAI: self.supports_seed = False

    def generate(
        self, system: str, user: str, max_tokens: int = 160, *,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float = 1.0, seed: int | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> Generation:
        from google.genai import types

        validate_sampling(max_tokens, temperature, top_k, top_p, seed)
        config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            config_kwargs["top_k"] = top_k
        if seed is not None:
            config_kwargs["seed"] = seed
        if output_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = gemini_output_schema(output_schema)

        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        # OpenAI (Responses API nie przyjmuje top_k ani seed):
        # if output_schema is not None:
        #     raise NotImplementedError(
        #         "Dla OpenAI dodaj osobną obsługę Structured Outputs"
        #     )
        # if top_k is not None or seed is not None:
        #     raise ValueError("OpenAI Responses API nie obsługuje top_k ani seed")
        # response = self.client.responses.create(
        #     model=self.model,
        #     instructions=system,
        #     input=user,
        #     max_output_tokens=max_tokens,
        #     temperature=0.0 if temperature is None else temperature,
        #     top_p=top_p,
        # )
        latency_s = time.perf_counter() - started

        generated_text = response.text
        # OpenAI: generated_text = response.output_text
        if not generated_text:
            raise RuntimeError("API nie zwróciło tekstu odpowiedzi")

        # OpenAI:
        # if response.status != "completed":
        #     raise RuntimeError(
        #         f"Generowanie nie zostało zakończone: "
        #         f"{response.status}, {response.incomplete_details}"
        #     )

        usage = response.usage_metadata
        # OpenAI: usage = response.usage
        return Generation(
            text=generated_text,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            # OpenAI: input_tokens=getattr(usage, "input_tokens", None),
            # OpenAI: output_tokens=getattr(usage, "output_tokens", None),
            latency_s=latency_s,
            model=self.model,
            mode="api",
        )


class LocalBackend:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
        )
        self.model.to(self.device).eval()

    def generate(
        self, system: str, user: str, max_tokens: int = 160, *,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float = 1.0, seed: int | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> Generation:
        validate_sampling(max_tokens, temperature, top_k, top_p, seed)
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature is not None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature is not None:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                # Transformers domyślnie stosuje top_k=50 przy próbkowaniu.
                top_k=0 if top_k is None else top_k,
            )
            if seed is not None:
                self.torch.manual_seed(seed)

        if output_schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
            parser = JsonSchemaParser(output_schema.model_json_schema())
            generation_kwargs["prefix_allowed_tokens_fn"] = (
                build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
            )

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        input_tokens = int(inputs["input_ids"].shape[1])
        new_ids = output[0, input_tokens:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        latency_s = time.perf_counter() - started

        return Generation(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=int(new_ids.shape[0]),
            latency_s=latency_s,
            model=self.model_name,
            mode="local",
        )


def make_backend() -> LLMBackend:
    mode = os.getenv("LLM_MODE", "local")
    if mode == "local":
        return LocalBackend()
    if mode == "api":
        return ApiBackend()
    raise ValueError("LLM_MODE musi mieć wartość 'local' albo 'api'")


def append_event(path: str, result: Generation, status: str = "ok") -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        **{k: v for k, v in asdict(result).items() if k != "text"},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def run_experiment(backend: LLMBackend) -> None:
    configurations = [
        ("A", None, 1.0),
        ("B", 0.3, 1.0),
        ("C", 0.8, 0.9),
        ("D", 1.2, 0.9),
    ]
    rows = []
    summaries = []
    for name, temperature, top_p in configurations:
        config_rows = []
        for run in range(1, 6):
            # Różne seed dla powtórzeń; taki sam seed dla numeru próby w B/C/D.
            seed = 42 + run if temperature is not None and getattr(backend, "supports_seed", True) else None
            result = backend.generate(
                system="Odpowiadaj krótko i jasno po polsku.",
                user="Podaj nazwę i jednozdaniowy opis aplikacji pomagającej studentowi planować naukę.",
                max_tokens=160,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            row = {
                "configuration": name,
                "run": run,
                "temperature": "greedy" if temperature is None else temperature,
                "top_p": top_p,
                "seed": seed,
                "text": result.text,
                "output_tokens": result.output_tokens,
                "latency_s": result.latency_s,
            }
            rows.append(row)
            config_rows.append(row)

        counts = Counter(normalized(row["text"]) for row in config_rows)
        lengths = [row["output_tokens"] for row in config_rows if row["output_tokens"] is not None]
        summary = {
            "configuration": name,
            "unique_responses": len(counts),
            "mean_output_tokens": statistics.fmean(lengths) if lengths else None,
            "mean_latency_s": statistics.fmean(row["latency_s"] for row in config_rows),
        }
        summaries.append(summary)
        print(
            f"{name}: unikalne={summary['unique_responses']}/5, "
            f"średnia długość={summary['mean_output_tokens']}, "
            f"średni czas={summary['mean_latency_s']:.2f} s"
        )

    with open("lab01b_sampling.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open("lab01b_sampling_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    backend = make_backend()
    if sys.argv[1:] == ["experiment"]:
        run_experiment(backend)
    else:
        result = backend.generate(
            system="Odpowiadaj krótko i jasno po polsku.",
            user="Wyjaśnij, czym jest token w modelu językowym.",
            max_tokens=160,
        )
        print(result.text)
        print(
            f"\nModel: {result.model} | tryb: {result.mode} | "
            f"wejście: {result.input_tokens} tokenów | "
            f"wyjście: {result.output_tokens} tokenów | "
            f"czas: {result.latency_s:.2f} s"
        )
        append_event("../../logs/test.jsonl", result)
