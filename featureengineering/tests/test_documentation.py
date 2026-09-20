"""Handbook safety and default report routes; never run real pruning or market-data checks."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

CODE_ROOT = Path(os.environ.get("FEA_DOC_TEST_ROOT", Path(__file__).resolve().parents[1]))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


doc = load("doc_under_test", CODE_ROOT / "fea/documentation.py")
TEMPLATE = ("# 手工说明\n不可覆盖\n<!-- FEA:DOC:factor-catalog:BEGIN -->\n旧字典\n"
            "<!-- FEA:DOC:factor-catalog:END -->\n手工尾注\n"
            "<!-- FEA:REPORTS:BEGIN -->\n<!-- FEA:REPORTS:END -->\n")


class HandbookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.readme = self.root / "README.md"
        self.readme.write_text(TEMPLATE, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_catalog_preserves_manual_text_and_is_idempotent(self):
        doc.update_section(self.root, "factor-catalog", "新字典\n")
        first = self.readme.read_bytes()
        self.assertEqual(first.decode(), TEMPLATE.replace("旧字典", "新字典"))
        doc.update_section(self.root, "factor-catalog", "新字典\n")
        self.assertEqual(first, self.readme.read_bytes())

    def test_reports_append_without_clobbering(self):
        doc.append_report(self.root, "adj-check", "first <check>", "report one")
        doc.append_report(self.root, "adj-check", "second", "report two")
        text = self.readme.read_text(encoding="utf-8")
        for value in ["report one", "report two", "不可覆盖", "旧字典", "first &lt;check&gt;"]:
            self.assertIn(value, text)
        self.assertEqual([p.name for p in self.root.rglob("*.md")], ["README.md"])

    def test_missing_duplicate_reversed_markers_refuse(self):
        begin = "<!-- FEA:DOC:factor-catalog:BEGIN -->"
        end = "<!-- FEA:DOC:factor-catalog:END -->"
        for bad in [TEMPLATE.replace(begin, ""), TEMPLATE + begin, end + "text" + begin]:
            self.readme.write_text(bad, encoding="utf-8")
            with self.assertRaises(ValueError):
                doc.update_section(self.root, "factor-catalog", "new")
            self.assertEqual(self.readme.read_text(encoding="utf-8"), bad)

    def test_missing_report_area_refuses(self):
        self.readme.write_text("manual only", encoding="utf-8")
        with self.assertRaises(ValueError):
            doc.append_report(self.root, "adj-check", "test", "body")
        self.assertEqual(self.readme.read_text(), "manual only")

    def test_control_marker_injection_refuses(self):
        with self.assertRaises(ValueError):
            doc.update_section(self.root, "factor-catalog", "<!-- FEA:REPORTS:END -->")
        self.assertEqual(self.readme.read_text(encoding="utf-8"), TEMPLATE)

    def test_missing_handbook_refuses(self):
        self.readme.unlink()
        with self.assertRaises(FileNotFoundError):
            doc.update_section(self.root, "factor-catalog", "new")
        self.assertFalse(self.readme.exists())

    def test_concurrent_reports_preserve_both(self):
        source = str(CODE_ROOT / "fea/documentation.py")
        code = ("import importlib.util; from pathlib import Path; "
                f"s=importlib.util.spec_from_file_location('d',{source!r}); "
                "d=importlib.util.module_from_spec(s); s.loader.exec_module(d); "
                f"d.append_report(Path({str(self.root)!r}), 'parallel', 'test', ")
        jobs = [subprocess.Popen([sys.executable, "-c", code + repr(msg) + ")"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for msg in ["parallel first", "parallel second"]]
        for job in jobs:
            out, err = job.communicate(timeout=20)
            self.assertEqual(job.returncode, 0, err.decode())
        body = self.readme.read_text(encoding="utf-8")
        self.assertIn("parallel first", body)
        self.assertIn("parallel second", body)

    def run_adj(self, extra=()):
        import pandas as pd
        adj = load("adj_test", CODE_ROOT / "scripts/check_adj_factor.py")
        adj.ROOT = self.root
        adj.KNOWN = self.root / "absent-baseline.json"
        with patch.object(adj, "run", return_value=([], pd.DataFrame())), \
             patch.dict(sys.modules, {"fea.documentation": doc}), \
             patch.object(sys, "argv", ["check_adj_factor.py", *extra]), \
             contextlib.redirect_stdout(io.StringIO()):
            return adj.main()

    def test_adj_default_appends_to_handbook(self):
        self.assertEqual(self.run_adj(), 0)
        self.assertIn("复权因子体检", self.readme.read_text(encoding="utf-8"))
        self.assertEqual(len(list(self.root.rglob("*.md"))), 1)

    def test_adj_explicit_handbook_overwrite_refuses(self):
        with self.assertRaises(ValueError):
            self.run_adj(["--out", str(self.readme)])
        self.assertEqual(self.readme.read_text(encoding="utf-8"), TEMPLATE)

    def run_prune(self, extra=()):
        prune = load("prune_test", CODE_ROOT / "scripts/prune_factors.py")
        prune.ROOT = self.root
        cfg = types.SimpleNamespace(factors_dir=self.root / "data", state_dir=self.root / "state")
        with patch.object(prune, "_specs", return_value={}), \
             patch("fea.config.load", return_value=cfg), \
             patch.dict(sys.modules, {"fea.documentation": doc}), \
             patch.object(sys, "argv", ["prune_factors.py", *extra]), \
             contextlib.redirect_stdout(io.StringIO()):
            return prune.main()

    def test_prune_preview_default_appends_only_report(self):
        self.assertEqual(self.run_prune(), 0)
        text = self.readme.read_text(encoding="utf-8")
        self.assertIn("计划删除", text)
        self.assertIn("不可覆盖", text)
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(len(list(self.root.rglob("*.md"))), 1)

    def test_prune_explicit_handbook_overwrite_refuses(self):
        with self.assertRaises(ValueError):
            self.run_prune(["--report", str(self.readme)])
        self.assertEqual(self.readme.read_text(encoding="utf-8"), TEMPLATE)


if __name__ == "__main__":
    unittest.main()
