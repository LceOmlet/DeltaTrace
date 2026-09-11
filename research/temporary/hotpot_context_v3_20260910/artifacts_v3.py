"""Read frozen predecessors without modifying their protocols or artifacts."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
V2=HERE.parent/'hotpot_fairness_20260910'
OLD=HERE.parent/'source_v2_gpu_20260910'
sys.path.insert(0,str(V2))
sys.path.insert(0,str(ROOT/'experiments/official'))
from common import full_cases,read_run,read_bytes,sha,byte_sha
