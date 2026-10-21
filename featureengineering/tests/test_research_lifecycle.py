"""Engineering gates never invent model contribution or untouched future evidence."""
import copy
import contextlib
import unittest
from unittest.mock import patch

from research.lifecycle import _check_evidence
from research.protocol import Protocol, digest


class ResearchLifecycleTests(unittest.TestCase):
    def evidence(self):
        evaluation = {"candidate_id": "candidate_v1", "calculation_run_id": "r_1",
                      "primary_horizons": [1, 5], "summary": [
                          {"horizon": 1, "days": 80, "rankic": .01},
                          {"horizon": 5, "days": 76, "rankic": -.002}],
                      "protocol_hash": "protocol", "definition_hash": "definition", "source_identity_hash": "source"}
        verification = {"candidate_id": "candidate_v1", "reference_run": "r_1", "passed": True}
        return evaluation, verification

    def check(self, evaluation, verification):
        with patch("research.lifecycle.read_evaluation", return_value=evaluation), \
             patch("research.lifecycle.read_verification", return_value=verification), \
             patch("research.lifecycle._current_evidence_identity"):
            return _check_evidence(None, "candidate_v1", "e_1", "v_1")

    def test_review_sufficient_samples_not_arbitrary_positive_alpha_rule(self):
        evidence = self.check(*self.evidence())
        self.assertFalse(evidence["validated_alpha"])
        self.assertIsNone(evidence["model_contribution"])
        self.assertIsNone(evidence["future_confirmation"])

    def test_engineering_failure_or_mismatched_reference_blocks(self):
        evaluation, verification = self.evidence()
        verification["passed"] = False
        with self.assertRaises(ValueError):
            self.check(evaluation, verification)
        verification["passed"] = True
        verification["reference_run"] = "another_run"
        with self.assertRaises(ValueError):
            self.check(evaluation, verification)

    def test_undefined_rankic_or_insufficient_days_blocks_stock_qualification(self):
        evaluation, verification = self.evidence()
        for edit in ({"days": 59}, {"rankic": None}):
            changed = copy.deepcopy(evaluation)
            changed["summary"][1].update(edit)
            with self.assertRaises(ValueError):
                self.check(changed, verification)

    def test_market_context_does_not_fake_cross_sectional_metrics(self):
        evaluation, verification = self.evidence()
        evaluation["is_market"] = True
        evaluation["summary"] = [{"horizon": 0, "days": 60, "rankic": None}]
        self.assertTrue(self.check(evaluation, verification)["engineering_passed"])
        evaluation["summary"][0]["days"] = 59
        with self.assertRaises(ValueError):
            self.check(evaluation, verification)

    def test_current_shared_code_configuration_and_protocol_gate(self):
        evaluation, verification = self.evidence()
        protocol = Protocol()
        calculation = {"common_sources": {"fea/mathx.py": "original"},
                       "objects": {"parent": {"recipe": "kernel=1", "version": 1}}}
        evaluation.update({"protocol_hash": protocol.identity, "computation_identity_hash": digest(calculation)})
        reference = {"candidate_id": "candidate_v1", "protocol_hash": protocol.identity,
                     "computation_identity": calculation, "computation_identity_hash": digest(calculation)}
        # Exercise the real gate while replacing only metadata reads/source
        # fingerprint construction. Engine prebuild/watermark are never used.
        with patch("research.lifecycle.read_evaluation", return_value=evaluation), \
             patch("research.lifecycle.read_verification", return_value=verification), \
             patch("research.lifecycle.read_run", return_value=(None, reference)), \
             patch("research.lifecycle.load_protocol", return_value=protocol) as protocol_read, \
             patch("research.lifecycle.spec_for_candidate", return_value=object()), \
             patch("research.lifecycle.candidate_registry", side_effect=lambda *args: contextlib.nullcontext({})), \
             patch("research.lifecycle.dependency_specs", return_value=[]), \
             patch("fea.engine.Engine") as engine, \
             patch("research.lifecycle.computation_identity", return_value=calculation) as identity:
            self.assertTrue(_check_evidence(None, "candidate_v1", "e_1", "v_1")["engineering_passed"])
            for change in ("source", "recipe"):
                revised = copy.deepcopy(calculation)
                if change == "source":
                    revised["common_sources"]["fea/mathx.py"] = "changed"
                else:
                    revised["objects"]["parent"]["recipe"] = "kernel=2"
                identity.return_value = revised
                with self.assertRaisesRegex(ValueError, "Current computation"):
                    _check_evidence(None, "candidate_v1", "e_1", "v_1")
            identity.return_value = calculation
            protocol_read.return_value = Protocol(block_days=40)
            with self.assertRaisesRegex(ValueError, "Current protocol"):
                _check_evidence(None, "candidate_v1", "e_1", "v_1")
            engine.return_value.prebuild.assert_not_called()
            engine.return_value.watermark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
