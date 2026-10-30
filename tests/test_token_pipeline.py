import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from token_pipeline import (
    Architecture,
    benchmark,
    generate_response,
    infer_text,
    load_model,
    load_chat_transitions,
    load_huggingface_texts,
    save_chat_data,
    tokenize,
    train,
)

ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "neura"


class TokenPipelineTests(unittest.TestCase):
    def test_token_training_routes_through_architecture(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".neur", encoding="utf-8") as architecture_file:
            architecture_file.write("# 1.0\n(INPUT) - 4\n(HIDDEN) - 3\n(OUTPUT) - 2\n")
            architecture_file.flush()
            architecture = Architecture.from_file(architecture_file.name)

        tokens = tokenize("Hello world, hello world!")
        state = train(tokens, architecture)
        result = benchmark(tokens, architecture, state)

        self.assertEqual(tokens[:2], ["hello", "world"])
        self.assertEqual(result["architecture"]["sizes"], [4, 3, 2])
        self.assertGreaterEqual(result["metrics"]["total_spikes"], 0)
        self.assertEqual(result["format"], "neuro-dynamics-benchmark-v1")
        self.assertEqual(result["training_backend"], "neura-c")
        self.assertGreater(result["layer_activity"][0]["spikes"], 0)
        json.dumps(result)

    def test_training_updates_and_serializes_synapses(self):
        architecture = Architecture(("INPUT", "OUTPUT"), (4, 3), 1.0)
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            architecture_file = directory_path / "test.neur"
            architecture_file.write_text(
                "# 1.0\n(INPUT) - 4\n(OUTPUT) - 3\n", encoding="utf-8"
            )
            initial_path = directory_path / "initial.model"
            trained_path = directory_path / "trained.model"
            detailed_log = directory_path / "training.neur"
            subprocess.run(
                [str(BACKEND), "--crfile", str(architecture_file), "--sfile", str(initial_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            initial = load_model(initial_path)
            state = train(
                tokenize("science science science"), architecture,
                model_path=initial_path, output_path=trained_path,
                log_path=detailed_log,
            )
            trained_bytes = trained_path.read_bytes()
            inference = infer_text(tokenize("science"), trained_path)
            self.assertEqual(trained_path.read_bytes(), trained_bytes)
            loaded = load_model(trained_path)
            detailed_log_size = detailed_log.stat().st_size
            subprocess.run(
                [str(BACKEND), "--rfile", str(trained_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(
            loaded.layers[1].connections[0], initial.layers[1].connections[0]
        )
        self.assertEqual(loaded.architecture.sizes, architecture.sizes)
        self.assertEqual(loaded.layers[1].connections[0], state["model"].layers[1].connections[0])
        self.assertGreater(state["total_spikes"], 0)
        self.assertGreater(state["weight_updates"], 0)
        self.assertGreater(detailed_log_size, 0)
        self.assertGreater(inference["spikes"], 0)
        self.assertEqual(inference["weight_updates"], 0)

    def test_chat_sidecar_and_response_generation(self):
        architecture = Architecture(("INPUT", "OUTPUT"), (4, 3), 1.0)
        transitions = {
            "hello": Counter({"world": 4}),
            "world": Counter({".": 4}),
        }
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "model.chat.json"
            save_chat_data(sidecar, architecture, transitions, token_count=5)
            loaded, metadata = load_chat_transitions(sidecar)

        response = generate_response(["hello"], loaded, max_tokens=4)
        self.assertEqual(metadata["format"], "neuro-dynamics-chat-v1")
        self.assertEqual(response, ("world.", 1.0))

    def test_legacy_n3_repairs_input_routes_and_learns(self):
        source_model = ROOT / "models" / "n3.model"
        architecture = load_model(source_model).architecture
        with tempfile.TemporaryDirectory() as directory:
            initial_path = Path(directory) / "legacy-n3.model"
            trained_path = Path(directory) / "trained-n3.model"
            shutil.copyfile(source_model, initial_path)
            state = train(
                tokenize("science neuron spikes learning science neuron spikes learning"),
                architecture,
                model_path=initial_path,
                output_path=trained_path,
            )

            self.assertTrue(trained_path.is_file())
            trained = load_model(trained_path)
            subprocess.run(
                [str(BACKEND), "--rfile", str(trained_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        input_layer = architecture.labels.index("INPUT")
        self.assertTrue(any(input_layer in layer.connections for layer in trained.layers))
        self.assertGreater(state["weight_updates"], 0)
        self.assertTrue(any(
            item["spikes"] > 0
            for item in state["traces"][0]
            if item["layer"] != "INPUT"
        ))

    def test_huggingface_loader_streams_bounded_text(self):
        rows = [{"article": "science"}, {"article": "research"}]
        loader = unittest.mock.Mock(return_value=rows)
        datasets_module = SimpleNamespace(load_dataset=loader)
        with patch.dict(sys.modules, {"datasets": datasets_module}):
            text = load_huggingface_texts(
                "ccdv/arxiv-summarization", None, "train", "article", 2, 10
            )

        self.assertEqual(text, "science\nres")
        loader.assert_called_once_with(
            "ccdv/arxiv-summarization", None, split="train", streaming=True
        )


if __name__ == "__main__":
    unittest.main()