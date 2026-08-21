import struct
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from parser import Event, read_log_file


class ParserTests(unittest.TestCase):
    def test_reads_complete_entries(self):
        payload = b"".join(
            struct.pack("<iiiff", event, neuron, layer, voltage, timestamp)
            for event, neuron, layer, voltage, timestamp in (
                (Event.EVENT_TYPE_SPIKE, 2, 1, -50.0, 0.1),
                (Event.EVENT_TYPE_WEIGTH_UPDATE, 3, 1, 0.25, 0.2),
            )
        )
        with tempfile.NamedTemporaryFile() as log_file:
            log_file.write(payload)
            log_file.flush()
            entries = read_log_file(log_file.name)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].event_type, Event.EVENT_TYPE_SPIKE)
        self.assertAlmostEqual(entries[1].data.neuron_u, 0.25)

    def test_ignores_truncated_tail(self):
        with tempfile.NamedTemporaryFile() as log_file:
            log_file.write(struct.pack("<iiiff", 0, 1, 0, -50.0, 0.1) + b"tail")
            log_file.flush()
            entries = read_log_file(log_file.name)

        self.assertEqual(len(entries), 1)


if __name__ == "__main__":
    unittest.main()
