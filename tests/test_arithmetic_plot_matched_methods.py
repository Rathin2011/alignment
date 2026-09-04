import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "toy/arithmetic/within_task"))

from plot_matched_methods import aggregate  # noqa: E402


class PlotAggregationTest(unittest.TestCase):
    def test_aggregate_uses_all_seeds(self):
        methods = ["regular_sft"]
        reports = []
        for value in (0.2, 0.4):
            reports.append({
                "methods": methods,
                "runs": [{"results": {"regular_sft": {
                    "heldout": {"accuracy": value},
                    "patch_accuracy": value + 0.1,
                    "atomic_accuracy": 1.0,
                    "history": [{"update": 0, "heldout": {"accuracy": value}}],
                }}}],
            })
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, report in enumerate(reports):
                path = Path(directory) / "{}.json".format(index)
                path.write_text(json.dumps(report))
                paths.append(path)
            result = aggregate(paths)
        self.assertAlmostEqual(
            result["final"]["regular_sft"]["heldout_accuracy"]["mean"], 0.3
        )
        self.assertEqual(
            result["final"]["regular_sft"]["seed_count"], 2
        )


if __name__ == "__main__":
    unittest.main()
