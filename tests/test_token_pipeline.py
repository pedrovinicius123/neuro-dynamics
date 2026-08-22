import json
import tempfile
import unittest
from pathlib import Path

from token_pipeline import Architecture, benchmark, tokenize, train


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
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()