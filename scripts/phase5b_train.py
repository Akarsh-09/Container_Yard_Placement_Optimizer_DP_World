"""Reuse the production train-only trace control without overwriting history."""
from pathlib import Path
import phase4
phase4.OUT=Path(__file__).resolve().parents[1]/'results/phase5b'
if __name__=='__main__':phase4.main()
