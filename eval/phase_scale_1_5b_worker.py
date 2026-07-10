"""Single-pair worker for phase_scale_1_5b.py.

Runs exactly one merge+perplexity+score job, writes the result to a JSON
file, and exits. Meant to be launched as its own subprocess by the driver
so that a hard crash (OOM, driver crash, mergekit segfault) on one pair
only kills that pair's process -- the driver's Python process, its CUDA
context, and its RAM never accumulate state across pairs and can't be
taken down by one bad pair.
"""

import json
import sys

from phase_scale_1_5b import run_pair


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
