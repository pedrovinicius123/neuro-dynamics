"""Train the project's spiking networks from text and scientific datasets."""
import argparse
import itertools
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

TOKEN_RE = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
MAX_LAYERS = 10
U_REST = -75.0


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
        if not sizes or len(sizes) > MAX_LAYERS or any(size <= 0 for size in sizes):
            raise ValueError(f"A arquitetura deve ter de 1 a {MAX_LAYERS} camadas válidas")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("A probabilidade de conexão deve estar entre 0 e 1")
        return cls(tuple(labels), tuple(sizes), probability)


@dataclass
class Neuron:
    current_timestamp: float = 0.0
    current_u: float = U_REST
    ref_count: float = 0.0
    last_spike_timestamp: float = 0.0
    idx: int = 0


@dataclass
class Layer:
    label: str
    neurons: list[Neuron]
    connections: dict[int, list[list[float]]] = field(default_factory=dict)


@dataclass
class SNNModel:
    layers: list[Layer]

    @property
    def architecture(self) -> Architecture:
        return Architecture(
            tuple(layer.label for layer in self.layers),
            tuple(len(layer.neurons) for layer in self.layers),
            0.0,
        )


def tokenize(text: str) -> list[str]:
    """Split Unicode text into stable, lowercase words and punctuation."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_source(source: str, max_chars: int = 250_000) -> str:
    """Load local, HTTP(S), or stdin data with a bounded payload."""
    if source == "-":
        return sys.stdin.read(max_chars)
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "neuro-dynamics/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read(max_chars).decode("utf-8", errors="replace")
    return Path(source).read_text(encoding="utf-8")[:max_chars]


def load_model(filename: str | Path) -> SNNModel:
    """Read the native little-endian model format consumed by neura.c."""
    data = memoryview(Path(filename).read_bytes())
    offset = 0

    def read(fmt: str):
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(data):
            raise ValueError("Arquivo de modelo truncado")
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        return values[0] if len(values) == 1 else values

    n_layers = read("<i")
    if not 0 < n_layers <= MAX_LAYERS:
        raise ValueError("Quantidade de camadas inválida no modelo")
    layers: list[Layer] = []
    pending_connections: list[tuple[int, int, int, list[list[float]]]] = []
    for _ in range(n_layers):
        label_size = read("<i")
        if not 0 < label_size <= 4096 or offset + label_size > len(data):
            raise ValueError("Tamanho de label inválido no modelo")
        label = bytes(data[offset:offset + label_size]).decode("utf-8")
        offset += label_size
        n_neurons = read("<i")
        layer_idx = read("<i")
        if not 0 < n_neurons <= 1_000_000 or layer_idx != len(layers):
            raise ValueError("Dimensões de camada inválidas no modelo")
        neurons = [Neuron(*read("<ffffi")) for _ in range(n_neurons)]
        for index, neuron in enumerate(neurons):
            neuron.idx = index
        layer = Layer(label, neurons)
        layers.append(layer)
        n_connections = read("<i")
        if not 0 <= n_connections < MAX_LAYERS:
            raise ValueError("Quantidade de conexões inválida no modelo")
        for _ in range(n_connections):
            source = read("<i")
            n_rows = read("<i")
            if not 0 <= source < n_layers or source == layer_idx or not 0 < n_rows <= 1_000_000:
                raise ValueError("Dimensão de conexão inválida no modelo")
            matrix = [[read("<f") for _ in range(n_neurons)] for _ in range(n_rows)]
            pending_connections.append((layer_idx, source, n_rows, matrix))
    if offset != len(data):
        raise ValueError("Dados extras no arquivo de modelo")
    for target, source, n_rows, matrix in pending_connections:
        if n_rows != len(layers[source].neurons):
            raise ValueError("Dimensões incompatíveis na matriz de conexão")
        layers[target].connections[source] = matrix
    return SNNModel(layers)


def _write_architecture(architecture: Architecture, filename: Path) -> None:
    lines = [f"# {architecture.connection_probability}"]
    lines.extend(f"({label}) - {size}" for label, size in zip(architecture.labels, architecture.sizes))
    filename.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(tokens: Iterable[str], architecture: Architecture,
          model_path: Path | None = None, architecture_path: Path | None = None,
          output_path: Path | None = None, backend: Path | None = None,
          log_path: Path | None = None, max_tokens: int = 20_000) -> dict:
    """Train through neura's native C LIF/STDP backend."""
    token_list = list(itertools.islice(tokens, max_tokens))
    if not token_list:
        raise ValueError("Nenhum token para treinar")
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for previous, current in itertools.pairwise(token_list):
        transitions[previous][current] += 1

    executable = Path(backend).resolve() if backend else Path(__file__).resolve().with_name("neura")
    if not executable.is_file():
        raise FileNotFoundError(f"Backend C não encontrado: {executable}. Execute make compile.")
    if not os.access(executable, os.X_OK):
        raise PermissionError(f"Backend C não executável: {executable}")

    with tempfile.TemporaryDirectory(prefix="neuro-dynamics-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        model_output = Path(output_path) if output_path else temporary_path / "trained.model"
        model_output.parent.mkdir(parents=True, exist_ok=True)
        detailed_log = Path(log_path) if log_path else None
        command = [str(executable)]
        if model_path is not None:
            command.extend(("--rfile", str(model_path)))
        else:
            if architecture_path is None:
                architecture_path = temporary_path / "network.neur"
                _write_architecture(architecture, architecture_path)
            command.extend(("--crfile", str(architecture_path)))
        command.extend(("--train-text", "--sfile", str(model_output)))
        if detailed_log is not None:
            detailed_log.parent.mkdir(parents=True, exist_ok=True)
            command.extend(("--log", str(detailed_log)))
        try:
            completed = subprocess.run(
                command,
                input="\n".join(token_list) + "\n",
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise RuntimeError(f"Não foi possível iniciar o backend C: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Backend C falhou ({completed.returncode}): {detail}")
        if not model_output.is_file():
            raise RuntimeError("Backend C não produziu o arquivo do modelo")

        network = load_model(model_output)
        metrics = {
            line.partition("=")[0]: line.partition("=")[2]
            for line in completed.stdout.splitlines()
            if "=" in line
        }
        trained_count = int(metrics.get("TRAINED_TOKENS", "0"))
        if trained_count != len(token_list):
            raise RuntimeError(
                f"Backend C treinou {trained_count} tokens; esperados {len(token_list)}"
            )
        spike_count = int(metrics.get("SPIKE_COUNT", "0"))
        weight_updates = int(metrics.get("WEIGHT_UPDATES", "0"))
        layer_counts = [
            int(value) for value in metrics.get("SPIKES_BY_LAYER", "").split(",") if value
        ]
        activity = [
            {"layer": label, "width": width,
             "spikes": layer_counts[index] if index < len(layer_counts) else 0}
            for index, (label, width) in enumerate(zip(architecture.labels, architecture.sizes))
        ]
        total_neurons = sum(architecture.sizes) * len(token_list)
        return {
            "token_count": len(token_list),
            "vocabulary_size": len(set(token_list)),
            "transitions": transitions,
            "traces": [activity],
            "total_spikes": spike_count,
            "firing_rate": spike_count / total_neurons if total_neurons else 0.0,
            "weight_updates": weight_updates,
            "model": network,
            "backend_output": completed.stdout.strip(),
        }


def benchmark(tokens: list[str], architecture: Architecture, state: dict,
              top_k: int = 5) -> dict:
    predictions = []
    transitions = state["transitions"]
    for token in tokens[-10:]:
        next_tokens = transitions.get(token, {})
        ranked = sorted(next_tokens.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        predictions.append({"input_token": token, "next_token_candidates": [item[0] for item in ranked]})
    return {
        "format": "neuro-dynamics-benchmark-v1",
        "training_backend": "neura-c",
        "tokenization": {"name": "unicode-ngram-spike-encoder", "lowercase": True},
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
            "weight_updates": state["weight_updates"],
        },
        "predictions": predictions,
        "layer_activity": state["traces"][-1] if state["traces"] else [],
    }


def save_chat_data(filename: str | Path, architecture: Architecture,
                   transitions: dict[str, Counter[str]], token_count: int) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": "neuro-dynamics-chat-v1",
        "training_backend": "neura-c",
        "architecture": {"labels": list(architecture.labels), "sizes": list(architecture.sizes)},
        "token_count": token_count,
        "transitions": {
            source: dict(sorted(next_tokens.items()))
            for source, next_tokens in sorted(transitions.items())
        },
    }
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary_name = output.name
            json.dump(data, output, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_chat_transitions(filename: str | Path) -> tuple[dict[str, Counter[str]], dict]:
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    transitions: dict[str, Counter[str]] = {}
    if data.get("format") == "neuro-dynamics-chat-v1":
        transitions = {
            source: Counter({token: int(count) for token, count in next_tokens.items()})
            for source, next_tokens in data.get("transitions", {}).items()
        }
    elif data.get("format") == "neuro-dynamics-benchmark-v1":
        for prediction in data.get("predictions", []):
            candidates = prediction.get("next_token_candidates", [])
            transitions[prediction["input_token"]] = Counter({
                token: len(candidates) - index
                for index, token in enumerate(candidates)
            })
    else:
        raise ValueError("Formato de dados de conversa desconhecido")
    if not transitions:
        raise ValueError("O arquivo não contém transições para conversar")
    return transitions, data


def generate_response(context: list[str], transitions: dict[str, Counter[str]],
                     max_tokens: int = 24) -> tuple[str, float] | None:
    start = next((token for token in reversed(context) if transitions.get(token)), None)
    if start is None:
        return None
    current = start
    response = []
    used_edges = set()
    first_confidence = 0.0
    for _ in range(max_tokens):
        if current not in transitions:
            break
        candidates = sorted(
            transitions[current].items(), key=lambda item: (-item[1], item[0])
        )
        total = sum(count for _, count in candidates)
        available = [item for item in candidates if (current, item[0]) not in used_edges]
        if not available:
            break
        next_token, count = available[0]
        if not response:
            first_confidence = count / total if total else 0.0
        used_edges.add((current, next_token))
        response.append(next_token)
        current = next_token
        if next_token in {".", "!", "?"} and len(response) >= 4:
            break
    if not response:
        return None
    text = ""
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}"}
    no_space_after = {"(", "[", "{"}
    previous = ""
    for token in response:
        if text and token not in no_space_before and previous not in no_space_after:
            text += " "
        text += token
        previous = token
    return text, first_confidence


def infer_text(tokens: Iterable[str], model_path: str | Path,
               backend: Path | None = None, log_path: Path | None = None) -> dict:
    token_list = list(tokens)
    if not token_list:
        return {"token_count": 0, "spikes": 0, "spikes_by_layer": []}
    executable = Path(backend).resolve() if backend else Path(__file__).resolve().with_name("neura")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Backend C não executável: {executable}. Execute make compile.")
    command = [str(executable), "--rfile", str(model_path), "--infer-text"]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(("--log", str(log_path)))
    completed = subprocess.run(
        command,
        input="\n".join(token_list) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Inferência C falhou ({completed.returncode}): {detail}")
    metrics = {
        line.partition("=")[0]: line.partition("=")[2]
        for line in completed.stdout.splitlines()
        if "=" in line
    }
    processed = int(metrics.get("INFERRED_TOKENS", "0"))
    if processed != len(token_list):
        raise RuntimeError(f"Backend C inferiu {processed} tokens; esperados {len(token_list)}")
    return {
        "token_count": processed,
        "spikes": int(metrics.get("SPIKE_COUNT", "0")),
        "weight_updates": int(metrics.get("WEIGHT_UPDATES", "0")),
        "spikes_by_layer": [
            int(value) for value in metrics.get("SPIKES_BY_LAYER", "").split(",") if value
        ],
    }


def run_chat(model_path: Path, chat_data_path: Path, backend: Path | None,
             feedback_path: Path, response_tokens: int = 24,
             log_path: Path | None = None) -> None:
    model = load_model(model_path)
    transitions, chat_data = load_chat_transitions(chat_data_path)
    saved_sizes = chat_data.get("architecture", {}).get("sizes")
    if saved_sizes and tuple(saved_sizes) != model.architecture.sizes:
        raise ValueError("Os dados de conversa não correspondem à arquitetura do modelo")

    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    context: list[str] = []
    print("Chat de diagnóstico. Respostas são continuações do corpus; /sair encerra, /reset limpa o contexto.")
    print(f"Modelo: {model_path} | Dados: {chat_data_path}")
    while True:
        try:
            prompt = input("Você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.casefold() in {"/sair", "/quit", "/exit"}:
            break
        if prompt.casefold() == "/reset":
            context.clear()
            print("Contexto limpo.")
            continue

        prompt_tokens = tokenize(prompt)
        context.extend(prompt_tokens)
        context = context[-64:]
        activity = infer_text(prompt_tokens, model_path, backend, log_path)
        response = generate_response(context, transitions, response_tokens)
        if response is None:
            response_text = "Não encontrei uma continuação para esse contexto no corpus de treino."
            confidence = 0.0
        else:
            response_text, confidence = response
            context.extend(tokenize(response_text))
            context = context[-64:]
        print(f"Neura> {response_text}")
        print(f"SNN C: {activity['spikes']} spikes; {activity['weight_updates']} atualizações STDP")
        print(f"Confiança da primeira transição: {confidence:.1%}")
        rating = input("Avaliação [c]orreta / [e]rrada / Enter para pular: ").strip().casefold()
        if rating in {"c", "e"}:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": str(model_path),
                "prompt": prompt,
                "response": response_text,
                "rating": "correct" if rating == "c" else "incorrect",
                "confidence": confidence,
                "spikes": activity["spikes"],
            }
            with feedback_path.open("a", encoding="utf-8") as feedback_file:
                feedback_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_huggingface_texts(dataset_name: str, config: str | None, split: str,
                           text_column: str, max_examples: int,
                           max_chars: int) -> str:
    """Stream a bounded amount of text from Hugging Face Datasets."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Instale as dependências com: pip install -r requirements.txt") from error
    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
    documents = []
    char_count = 0
    for row in itertools.islice(dataset, max_examples):
        text = row.get(text_column)
        if not isinstance(text, str):
            raise ValueError(f"A coluna {text_column!r} não contém texto")
        remaining = max_chars - char_count
        if remaining <= 0:
            break
        text = text[:remaining]
        documents.append(text)
        char_count += len(text)
    return "\n".join(documents)


def _model_paths(args) -> list[tuple[Path, Architecture, Path | None, Path | None]]:
    if args.model:
        input_path = Path(args.model)
        output_path = Path(args.model_output) if args.model_output else input_path
        model = load_model(input_path)
        return [(output_path, model.architecture, input_path, None)]
    if args.architecture:
        architecture_path = Path(args.architecture)
        architecture = Architecture.from_file(str(architecture_path))
        path = Path(args.model_output) if args.model_output else Path(args.models_dir) / f"{architecture_path.stem}.model"
        if path.exists():
            loaded_model = load_model(path)
            if loaded_model.architecture.sizes != architecture.sizes:
                raise ValueError("A arquitetura não corresponde ao modelo existente")
            return [(path, architecture, path, None)]
        return [(path, architecture, None, architecture_path)]
    paths = sorted(Path(args.models_dir).glob("*.model"))
    if not paths:
        raise ValueError(f"Nenhum .model encontrado em {args.models_dir}; informe --architecture")
    loaded: list[tuple[Path, Architecture, Path | None, Path | None]] = []
    for path in paths:
        model = load_model(path)
        loaded.append((path, model.architecture, path, None))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e testa redes SNN com texto")
    parser.add_argument("source", nargs="?", help="Arquivo UTF-8, URL ou - para stdin")
    parser.add_argument("--architecture", help="Arquivo .neur para criar ou selecionar um modelo")
    parser.add_argument("--model", help="Treina um modelo .model existente")
    parser.add_argument("--model-output", help="Destino do modelo serializado")
    parser.add_argument("--models-dir", default="models", help="Pasta dos modelos .model existentes")
    parser.add_argument("--output", default="benchmark.json", help="JSON de métricas")
    parser.add_argument("--dataset", default="ccdv/arxiv-summarization", help="Dataset científico Hugging Face")
    parser.add_argument("--dataset-config", help="Configuração do dataset")
    parser.add_argument("--split", default="train", help="Split do dataset")
    parser.add_argument("--text-column", default="article", help="Coluna textual do dataset")
    parser.add_argument("--max-examples", type=int, default=100, help="Documentos do dataset")
    parser.add_argument("--max-chars", type=int, default=250_000, help="Limite de caracteres carregados")
    parser.add_argument("--max-tokens", type=int, default=20_000, help="Limite de tokens treinados por modelo")
    parser.add_argument("--backend", type=Path, help="Executável C neura")
    parser.add_argument("--log", type=Path, help="Arquivo binário detalhado de spikes/STDP")
    parser.add_argument("--chat", action="store_true", help="Iniciar conversa interativa de diagnóstico")
    parser.add_argument("--chat-data", type=Path, help="Sidecar de transições ou benchmark JSON")
    parser.add_argument("--feedback-file", type=Path, default=Path("chat_feedback.jsonl"))
    parser.add_argument("--response-tokens", type=int, default=24)
    args = parser.parse_args()

    if args.chat:
        if not args.model:
            raise SystemExit("--chat requer --model models/<nome>.model")
        model_path = Path(args.model)
        sidecar_path = Path(f"{model_path}.chat.json")
        chat_data_path = args.chat_data or (
            sidecar_path if sidecar_path.exists() else Path(args.output)
        )
        if not chat_data_path.is_file():
            raise SystemExit(
                f"Dados de conversa não encontrados: {chat_data_path}. "
                "Treine o modelo novamente ou informe --chat-data."
            )
        run_chat(
            model_path,
            chat_data_path,
            args.backend,
            args.feedback_file,
            args.response_tokens,
            args.log,
        )
        return

    if args.source:
        text = load_source(args.source, args.max_chars)
    else:
        text = load_huggingface_texts(
            args.dataset, args.dataset_config, args.split,
            args.text_column, args.max_examples, args.max_chars,
        )
    tokens = tokenize(text)[:args.max_tokens]
    if not tokens:
        raise SystemExit("Nenhum token encontrado na fonte de dados")

    results = []
    for path, architecture, model_path, architecture_path in _model_paths(args):
        state = train(
            tokens,
            architecture,
            model_path=model_path,
            architecture_path=architecture_path,
            output_path=path,
            backend=args.backend,
            log_path=args.log,
            max_tokens=args.max_tokens,
        )
        chat_data_path = Path(f"{path}.chat.json")
        save_chat_data(chat_data_path, architecture, state["transitions"], state["token_count"])
        results.append(benchmark(tokens[:state["token_count"]], architecture, state))
        print(f"Treinado: {path} ({state['token_count']} tokens)")
        print(f"Dados de conversa: {chat_data_path}")
        if state["backend_output"]:
            print(state["backend_output"])
    output = results[0] if len(results) == 1 else {
        "format": "neuro-dynamics-training-v1",
        "training_backend": "neura-c",
        "models": results,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([result["metrics"] for result in results], ensure_ascii=False))


if __name__ == "__main__":
    main()