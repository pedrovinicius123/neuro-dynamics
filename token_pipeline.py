"""Token training and reproducible SNN benchmark adapter.

The module deliberately uses only the Python standard library so a dataset can
be fetched in a clean environment. Its JSON output is suitable for comparing
this project with an external model without parsing terminal output.
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Architecture:
    labels: tuple[str, ...]
    sizes: tuple[int, ...]
    connection_probability: float

    @classmethod
    def from_file(cls, filename: str) -> "Architecture":
        labels = []
        sizes = []
        probability = 0.0
        for raw_line in Path(filename).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                probability = float(line[1:].strip())
                continue
            match = re.fullmatch(r"\(([^)]+)\)\s*-\s*(\d+)", line)
            if not match:
                raise ValueError(f"Linha inválida na arquitetura: {raw_line}")
            labels.append(match.group(1))
            sizes.append(int(match.group(2)))
        if not sizes or sizes[0] <= 0:
            raise ValueError("A arquitetura precisa de uma camada de entrada")
        return cls(tuple(labels), tuple(sizes), probability)


def load_source(source: str, max_chars: int = 250_000) -> str:
    """Load local, HTTP(S), or stdin data with a bounded payload."""
    if source == "-":
        return sys.stdin.read(max_chars)
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "neuro-dynamics/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read(max_chars).decode("utf-8", errors="replace")
    return Path(source).read_text(encoding="utf-8")[:max_chars]


def tokenize(text: str) -> list[str]:
    """Create a stable, inspectable token stream for training and inference."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _layer_spikes(values: list[float], width: int, layer_index: int) -> list[int]:
    spikes = [0] * width
    for neuron in range(width):
        total = 0.0
        for source, value in enumerate(values):
            weight = ((source * 31 + neuron * 17 + layer_index * 13) % 23 - 11) / 11.0
            total += value * weight
        if total / max(1, len(values)) > 0.12:
            spikes[neuron] = 1
    return spikes


def architecture_trace(token: str, architecture: Architecture) -> list[dict]:
    """Encode one token and return activity at every architecture layer."""
    values = [0.0] * architecture.sizes[0]
    for char in token:
        values[ord(char) % len(values)] += 1.0
    trace = []
    for layer_index, width in enumerate(architecture.sizes):
        if layer_index:
            values = [float(spike) for spike in _layer_spikes(values, width, layer_index)]
        else:
            values = [1.0 if value else 0.0 for value in values]
        spikes = sum(int(value > 0) for value in values)
        trace.append({"layer": architecture.labels[layer_index], "width": width, "spikes": spikes})
    return trace


def train(tokens: Iterable[str], architecture: Architecture) -> dict:
    token_list = list(tokens)
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for previous, current in zip(token_list, token_list[1:]):
        transitions[previous][current] += 1
    traces = [architecture_trace(token, architecture) for token in token_list]
    total_spikes = sum(layer["spikes"] for trace in traces for layer in trace)
    total_neurons = sum(architecture.sizes) * max(1, len(token_list))
    return {
        "token_count": len(token_list),
        "vocabulary_size": len(set(token_list)),
        "transitions": transitions,
        "traces": traces,
        "total_spikes": total_spikes,
        "firing_rate": total_spikes / total_neurons if total_neurons else 0.0,
    }


def benchmark(tokens: list[str], architecture: Architecture, state: dict, top_k: int = 5) -> dict:
    predictions = []
    transitions = state["transitions"]
    for token in tokens[-10:]:
        next_tokens = transitions.get(token, {})
        ranked = sorted(next_tokens.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        predictions.append({"input_token": token, "next_token_candidates": [item[0] for item in ranked]})
    return {
        "format": "neuro-dynamics-benchmark-v1",
        "tokenization": {"name": "unicode-word-punctuation", "lowercase": True},
        "architecture": {
            "labels": list(architecture.labels),
            "sizes": list(architecture.sizes),
            "connection_probability": architecture.connection_probability,
        },
        "metrics": {
            "token_count": state["token_count"],
            "vocabulary_size": state["vocabulary_size"],
            "total_spikes": state["total_spikes"],
            "firing_rate": state["firing_rate"],
        },
        "predictions": predictions,
        "layer_activity": state["traces"][-1] if state["traces"] else [],
    }


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Treina e avalia uma arquitetura SNN com texto real")
    argument_parser.add_argument("source", help="URL http(s), arquivo UTF-8 ou - para stdin")
    argument_parser.add_argument("--architecture", required=True, help="Arquivo .neur")
    argument_parser.add_argument("--output", default="benchmark.json", help="JSON de benchmark")
    argument_parser.add_argument("--max-chars", type=int, default=250_000)
    args = argument_parser.parse_args()

    text = load_source(args.source, args.max_chars)
    tokens = tokenize(text)
    architecture = Architecture.from_file(args.architecture)
    state = train(tokens, architecture)
    result = benchmark(tokens, architecture, state)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
