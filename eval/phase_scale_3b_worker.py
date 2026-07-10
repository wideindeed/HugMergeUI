"""Single-pair worker for phase_scale_3b.py. Same subprocess-isolation
design as phase_scale_1_5b_worker.py - one merge+perplexity+score job per
process so a hard crash on one pair can't take down the driver or leak
GPU/RAM state into the next pair.
"""

import json
import sys

from phase_scale_3b import run_pair


def main() -> None:
    spec = json.loads(sys.argv[1])
    result = run_pair(
        spec["family_name"],
        spec["base_repo"],
        spec["name_a"],
        spec["repo_a"],
        spec["name_b"],
        spec["repo_b"],
    )
    with open(spec["out_path"], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
