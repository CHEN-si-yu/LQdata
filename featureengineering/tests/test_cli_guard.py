import unittest
from main import _mutating_cli

class CliGuardTests(unittest.TestCase):
    def test_default_run_and_rebuild_are_writers(self):
        for args in (["python","main.py"],["python","main.py","run"],["python","main.py","rebuild"],["python","scripts/backfill_history.py","--jobs","1"]):
            self.assertTrue(_mutating_cli(args),args)

    def test_read_only_and_sandbox_are_not_production_writers(self):
        for args in (["python","main.py","check"],["python","main.py","rebuild","--dry-run"],["python","main.py","run","--sandbox","tmp"],["python","other.py"]):
            self.assertFalse(_mutating_cli(args),args)
