#!/usr/bin/env python3
"""
draftbench - Benchmark token throughput on OpenAI-compatible endpoints.

Measures time-to-first-token (TTFT), tokens-per-second (TPS), and total
generation time by streaming completions and timing each SSE chunk.
Useful for demonstrating the throughput gains of speculative decoding.

Usage:
    python bench.py --base-url http://localhost:8000/v1 --model my-model
    python bench.py --base-url http://localhost:8000/v1 --model my-model \
                    --compare-url http://localhost:8001/v1 --compare-model my-model-spec
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import requests


@dataclass
class RequestMetrics:
    ttft: float  # seconds
    total_time: float  # seconds
    prompt_tokens: int
    completion_tokens: int
    tps: float  # completion tokens / generation time after first token


@dataclass
class RunSummary:
    label: str
    metrics: list[RequestMetrics] = field(default_factory=list)

    def stat(self, key: str):
        vals = [getattr(m, key) for m in self.metrics]
        if not vals:
            return {}
        vals.sort()
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p95": vals[int(len(vals) * 0.95)] if len(vals) >= 2 else vals[-1],
        }

    def table(self):
        rows = []
        for key, unit in [
            ("ttft", "s"),
            ("tps", "tok/s"),
            ("total_time", "s"),
            ("completion_tokens", "tok"),
        ]:
            s = self.stat(key)
            if not s:
                continue
            rows.append(
                f"  {key:<20s}  "
                f"min={s['min']:>8.2f}{unit}  "
                f"median={s['median']:>8.2f}{unit}  "
                f"mean={s['mean']:>8.2f}{unit}  "
                f"p95={s['p95']:>8.2f}{unit}  "
                f"max={s['max']:>8.2f}{unit}"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# SSE streaming request
# ---------------------------------------------------------------------------

def stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> RequestMetrics:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    token_count = 0
    first_token_time = None
    prompt_tokens = 0
    completion_tokens_reported = 0

    t_start = time.perf_counter()

    # Retry on 503 (server still loading model)
    max_retries = 5
    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        if resp.status_code == 503 and attempt < max_retries - 1:
            resp.close()
            time.sleep(3)
            continue
        resp.raise_for_status()
        break

    with resp:
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data: "):
                continue
            data = raw_line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # usage block (often in the final chunk)
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens_reported = usage.get(
                    "completion_tokens", completion_tokens_reported
                )

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if content or reasoning:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if content:
                    token_count += 1

    t_end = time.perf_counter()

    total_time = t_end - t_start
    ttft = (first_token_time - t_start) if first_token_time else total_time
    gen_time = t_end - first_token_time if first_token_time else 0.0

    final_tokens = completion_tokens_reported or token_count
    tps = (final_tokens - 1) / gen_time if gen_time > 0 and final_tokens > 1 else 0.0

    return RequestMetrics(
        ttft=ttft,
        total_time=total_time,
        prompt_tokens=prompt_tokens,
        completion_tokens=final_tokens,
        tps=tps,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    # Code - very predictable syntax, high acceptance rate
    "Write a Python class for a binary search tree with insert, delete, search, and in-order traversal methods. Include docstrings and type hints.",

    # JSON - highly structured, very predictable
    "Generate a JSON array of 10 fictional users. Each user should have: id, firstName, lastName, email, age, address (with street, city, zipCode), and a list of 3 hobbies.",

    # Lists - repetitive patterns, good acceptance rate
    "List the 20 largest countries by area. For each, provide: name, capital, population, and area in square kilometers. Format as a numbered list.",
]


def build_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bench(
    label: str,
    base_url: str,
    model: str,
    prompts: list[str],
    runs: int,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> RunSummary:
    summary = RunSummary(label=label)
    total = runs * len(prompts)
    idx = 0

    for run_i in range(runs):
        for prompt in prompts:
            idx += 1
            sys.stdout.write(f"\r  [{label}] request {idx}/{total} ...")
            sys.stdout.flush()
            try:
                m = stream_chat_completion(
                    base_url=base_url,
                    model=model,
                    messages=build_messages(prompt),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=api_key,
                )
                summary.metrics.append(m)
                sys.stdout.write(
                    f"\r  [{label}] request {idx}/{total}  "
                    f"ttft={m.ttft:.3f}s  tps={m.tps:.1f}  "
                    f"tokens={m.completion_tokens}\n"
                )
            except Exception as exc:
                sys.stdout.write(
                    f"\r  [{label}] request {idx}/{total}  ERROR: {exc}\n"
                )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark token throughput on OpenAI-compatible endpoints."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the primary endpoint (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument("--model", required=True, help="Model name for the primary endpoint")
    parser.add_argument("--api-key", default=None, help="API key (primary endpoint)")

    parser.add_argument(
        "--compare-url",
        default=None,
        help="Base URL of a second endpoint to compare against",
    )
    parser.add_argument("--compare-model", default=None, help="Model name for the compare endpoint")
    parser.add_argument("--compare-api-key", default=None, help="API key (compare endpoint)")
    parser.add_argument("--compare-label", default=None, help="Display label for the compare endpoint")

    parser.add_argument("--label", default=None, help="Display label for the primary endpoint")
    parser.add_argument("--runs", type=int, default=1, help="Number of full passes over the prompt set (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate per request (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--prompt", action="append", help="Custom prompt(s). Can be specified multiple times. Overrides built-in prompts.")

    args = parser.parse_args()

    prompts = args.prompt if args.prompt else PROMPTS

    label_a = args.label or "baseline"
    print(f"\n{'='*70}")
    print(f"  draftbench - OpenAI-compatible endpoint benchmark")
    print(f"{'='*70}")
    print(f"  Endpoint : {args.base_url}")
    print(f"  Model    : {args.model}")
    print(f"  Prompts  : {len(prompts)}")
    print(f"  Runs     : {args.runs}")
    print(f"  MaxTok   : {args.max_tokens}")
    print(f"  Temp     : {args.temperature}")
    print(f"{'='*70}\n")

    summary_a = run_bench(
        label=label_a,
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        runs=args.runs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
    )

    summary_b = None
    if args.compare_url:
        label_b = args.compare_label or "speculative"
        compare_model = args.compare_model or args.model
        print()
        summary_b = run_bench(
            label=label_b,
            base_url=args.compare_url,
            model=compare_model,
            prompts=prompts,
            runs=args.runs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            api_key=args.compare_api_key,
        )

    # ---- results ----
    print(f"\n{'='*70}")
    print(f"  RESULTS: {label_a}")
    print(f"{'='*70}")
    print(summary_a.table())

    if summary_b:
        label_b = summary_b.label
        print(f"\n{'='*70}")
        print(f"  RESULTS: {label_b}")
        print(f"{'='*70}")
        print(summary_b.table())

        # delta
        mean_tps_a = summary_a.stat("tps")["mean"]
        mean_tps_b = summary_b.stat("tps")["mean"]
        mean_ttft_a = summary_a.stat("ttft")["mean"]
        mean_ttft_b = summary_b.stat("ttft")["mean"]

        if mean_tps_a > 0:
            speedup = (mean_tps_b - mean_tps_a) / mean_tps_a * 100
        else:
            speedup = 0

        print(f"\n{'='*70}")
        print(f"  COMPARISON ({label_b} vs {label_a})")
        print(f"{'='*70}")
        print(f"  Mean TPS   : {mean_tps_a:.2f} -> {mean_tps_b:.2f}  ({speedup:+.1f}%)")
        print(f"  Mean TTFT  : {mean_ttft_a:.3f}s -> {mean_ttft_b:.3f}s")
        print()

    print()


if __name__ == "__main__":
    main()
