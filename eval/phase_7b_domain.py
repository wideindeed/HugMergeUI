"""Follow-up to phase_7b_mistral.py: generic-prose perplexity showed no
correlation between drift_magnitude/conflict and perplexity across the 4
Mistral-7B pairs. Diagnosis: generic prose can't see damage to a specific
fine-tune's specialty (a math+code merge can stay fluent describing coral
reefs while losing math/code ability). This reruns perplexity only, on
domain-matched eval text, and compares each merge against its unmerged
solo fine-tune's perplexity on the same domain - the delta is the real
damage signal, not the raw perplexity number.

Re-merging is cheap (models already cached, no downloads; ~1min GPU merge
each). Conflict/drift values from phase_7b_mistral_results.json are reused
unchanged since they don't depend on eval text.
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_texts_domain import MATH_EVAL_TEXTS, CODE_EVAL_TEXTS  # noqa: E402

FINETUNES = {
    "openhermes": "teknium/OpenHermes-2.5-Mistral-7B",
    "dolphin": "cognitivecomputations/dolphin-2.1-mistral-7b",
    "metamath": "abacusai/Fewshot-Metamath-OrcaVicuna-Mistral",
    "code": "ajibawa-2023/Code-Mistral-7B",
}

CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "phase_7b_mistral"
MERGED_DIR = Path(__file__).resolve().parent / "merged"
RESULTS_PATH = Path(__file__).resolve().parent / "phase_7b_domain_results.json"

TASKS = [
    dict(pair="openhermes+metamath", name_a="openhermes", name_b="metamath",
         domain="math", texts=MATH_EVAL_TEXTS, solo_name="metamath"),
    dict(pair="dolphin+code", name_a="dolphin", name_b="code",
         domain="code", texts=CODE_EVAL_TEXTS, solo_name="code"),
    dict(pair="metamath+code", name_a="metamath", name_b="code",
         domain="math", texts=MATH_EVAL_TEXTS, solo_name="metamath"),
    dict(pair="metamath+code", name_a="metamath", name_b="code",
         domain="code", texts=CODE_EVAL_TEXTS, solo_name="code"),
]


def mean_perplexity(model_path: str, texts: list[str], device: str = "cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = model(input_ids, labels=input_ids)
            losses.append(out.loss.item())

    del model
    torch.cuda.empty_cache()
    return math.exp(sum(losses) / len(losses))


def main() -> None:
    results: list[dict] = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"resuming: {len(results)} task(s) already completed", flush=True)

    done_keys = {(r["pair"], r["domain"]) for r in results}
    solo_baselines: dict[str, float] = {}

    def get_solo_ppl(repo: str, texts: list[str], label: str) -> float:
        key = f"{repo}::{label}"
        if key in solo_baselines:
            return solo_baselines[key]
        print(f"[solo] {repo} on {label}...", flush=True)
        ppl = mean_perplexity(repo, texts)
        solo_baselines[key] = ppl
        return ppl

    for t in TASKS:
        key = (t["pair"], t["domain"])
        if key in done_keys:
            continue

        pair_name = f"mistral7b__{t['name_a']}+{t['name_b']}"
        merged_path = MERGED_DIR / pair_name
        config_path = CONFIGS_DIR / f"{pair_name}.yaml"

        try:
            if not merged_path.exists():
                print(f"[{pair_name}] merging...", flush=True)
                subprocess.run(["mergekit-yaml", str(config_path), str(merged_path), "--cuda"], check=True)

            merged_ppl = mean_perplexity(str(merged_path.resolve()), t["texts"])
            solo_repo = FINETUNES[t["solo_name"]]
            solo_ppl = get_solo_ppl(solo_repo, t["texts"], t["domain"])

            result = {
                "pair": t["pair"],
                "domain": t["domain"],
                "merged_perplexity": merged_ppl,
                "solo_perplexity": solo_ppl,
                "degradation": merged_ppl - solo_ppl,
                "degradation_pct": (merged_ppl - solo_ppl) / solo_ppl * 100,
            }
            print(f"[{pair_name}/{t['domain']}] done: {result}", flush=True)
            results.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
        except Exception as e:
            print(f"[{pair_name}/{t['domain']}] FAILED: {e}", flush=True)
        finally:
            shutil.rmtree(merged_path, ignore_errors=True)

    print("\n=== ALL DOMAIN RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
