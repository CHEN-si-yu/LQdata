#!/usr/bin/env python3
"""Run V33 four-factor LambdaRank holdout model fitting or accounting."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--fit-only",action="store_true")
    args=parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/"scheduler.py"),"--jobs",str(args.jobs)],check=True)
    if not args.fit_only:
        subprocess.run([sys.executable,str(ROOT/"accounting.py")],check=True)
if __name__=="__main__": main()
