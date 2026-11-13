import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unittest

from subprojects.taris_variants.audit_sizing import summarize_rows


class SizingAuditTests(unittest.TestCase):
    def test_false_safe_is_separate_from_unscored_risk(self):
        def row(passed, scored, true, estimate):
            return {"profile": "adaptive", "variant": "hybrid", "scored": scored,
                    "estimated_passed": passed, "true_exceeded": true > 0.00004,
                    "true_window_loss_upper": true, "estimated_window_loss": estimate}
        rows = [row(True, True, 0.001, 0.00001), row(False, False, 0.001, None),
                row(False, True, 0.00001, 0.0001)]
        summary = summarize_rows(rows)[0]
        self.assertEqual(summary["admitted"], 3)
        self.assertEqual(summary["true_parameter_cap_exceeded"], 2)
        self.assertEqual(summary["false_safe_among_estimated_passed"], 1)
        self.assertEqual(summary["true_feasible_but_estimated_not_passed"], 1)


if __name__ == "__main__":
    unittest.main()
