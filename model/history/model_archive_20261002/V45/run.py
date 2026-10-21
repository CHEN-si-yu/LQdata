"""V45: replay the locally stored four-seed percentile-rank pool."""
from pathlib import Path
import subprocess, sys
UNIT = Path(__file__).resolve().parent
raise SystemExit(subprocess.call([sys.executable, "-B", str(UNIT / "analysis.py"), "--from-scores"], cwd=UNIT))
