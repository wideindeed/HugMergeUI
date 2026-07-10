"""Single-pair worker for phase_arch_phi2.py. Same subprocess-isolation
design as phase_scale_3b_worker.py / phase_hierarchy_worker.py.
"""

import json
import sys

from phase_arch_phi2 import run_pair


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
