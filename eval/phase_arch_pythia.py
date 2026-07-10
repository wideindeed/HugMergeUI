"""Broadening axis (b), take three, second attempt. OLMo-2-1B
(eval/phase_arch_olmo.py) was tried first but is a dead end for a new
reason: its config's model_type is "olmo2", which the pinned transformers
in this venv (4.46.3) doesn't recognize at all (Olmo2 support landed in
4.47). Upgrading transformers isn't a call to make lightly here - it's
the actual production dependency backing mergekit and the conflict
engine, not something scoped to eval scripts, so bumping it just to
chase a third architecture point isn't worth the risk. Abandoned, script
left on disk as a record (matches the phi-2 precedent in Round Eight).

Pivoted to Pythia-1.4B (EleutherAI, Apache 2.0, model_type "gpt_neox" -
a real architectural difference from Qwen2/StableLM2/Llama-style models,
natively supported for a very long time, fully ungated). Community
fine-tune ecosystem here commonly resizes vocab independently (same
failure mode as StableLM's "brief" in Round Eight) - several candidates
were checked via config.json before picking three with a vocab_size that
actually matches the base (50304):

  - sft: Leogrin/eleuther-pythia1.4b-hh-sft (Anthropic-HH SFT)
  - dpo: Leogrin/eleuther-pythia1.4b-hh-dpo (Anthropic-HH DPO, same
    author/lineage as sft, matched pair)
  - jokes: AlekseyKorshuk/pythia-1.4b-deduped-jokes (community, joke
    generation - general-purpose stylistic fine-tune, not a domain
    specialist, playing the same role "brief" played for StableLM)

3 non-trivial pairs + 1 self-anchor, 4 points. fp32, matching the main
family methodology. Same subprocess-per-pair + resume-from-checkpoint +
clean-retry design as phase_arch_stablelm.py.
"""

import itertools
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Root cause of the repeated download stalls (first diagnosed as IPv6-only,
# then recurred on a plain IPv4 CloudFront IP too): huggingface_hub's plain
# requests-based http_get has no read timeout by default, so a connection
# that silently dies mid-transfer (remote sends FIN, client sits in
# CLOSE_WAIT) just hangs forever instead of raising and retrying/resuming.
# This caps each chunk read so a dead connection surfaces as a timeout that
# huggingface_hub's own retry/backoff logic can act on.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

import torch
import yaml
import huggingface_hub.file_download as _hf_file_download
from huggingface_hub import snapshot_download

# There is no real HF_HUB_DISABLE_SYMLINKS env var in this huggingface_hub
# version (only HF_HUB_DISABLE_SYMLINKS_WARNING, which just silences the
# message) - confirmed by grepping the installed package. The actual check
# is are_symlinks_supported(), which runs a one-off dummy symlink test per
# cache dir and caches the result for the process's lifetime. That dummy
# test intermittently succeeds on this machine even though real symlink
# creation for an actual downloaded file then fails with WinError 1314
# (privilege not held) - a flaky Windows symlink-privilege quirk, not
# something worth chasing further. Forcing the function to always report
# "unsupported" makes huggingface_hub fall back to plain file copies
# unconditionally, sidestepping the flakiness entirely.
_hf_file_download.are_symlinks_supported = lambda cache_dir=None: False
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.conflict.engine import score_model_pair  # noqa: E402

from eval_texts import EVAL_TEXTS  # noqa: E402


def force_ipv4() -> None:
    """The huggingface_hub download of the base model's safetensors shard
    repeatedly stalled indefinitely (silent, no error - connection sat in
    CLOSE_WAIT with no retry) when routed over IPv6 to this CDN's AWS
    backend. Plain curl -4 to the same signed URL pulled the file cleanly
    at ~30MB/s, curl -6 couldn't even resolve one of the redirect hosts.
    Forcing IPv4-only resolution for this eval process sidesteps the
    routing issue without touching any shared/production dependency.
    """
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only_getaddrinfo


def prefetch_all_repos() -> None:
    repos = {BASE, *FINETUNES.values()}
    for repo in repos:
        print(f"[prefetch] {repo}...", flush=True)
        snapshot_download(repo, ignore_patterns=["*.bin", "*.msgpack", "*.h5"])


def prepare_local_bin_dir(repo_id: str) -> str:
    """mergekit's gpt_neox architecture definition requires attention.bias/
    masked_bias/rotary_emb.inv_freq, which are non-persistent PyTorch buffers
    that HF's safetensors conversion correctly drops. Any repo that publishes
    both formats (the base repo, and the sft fine-tune) fails for this reason,
    since mergekit's ModelReference.local_path() always prefers
    model.safetensors within a directory when both are present - this fails
    even on a base+base self-merge. Building a directory that contains ONLY
    the legacy pytorch_model.bin (plus config/tokenizer) forces mergekit to
    fall back to the format that actually has the tensors it needs. Safe to
    call for every repo in the merge, including ones that only ever publish
    .bin (dpo, jokes) - it's a no-op copy in that case. score_model_pair()
    (unlike mergekit's TIES merge) skips tensors missing from any model
    rather than erroring on them, so it keeps using the plain repo id.
    """
    dest = Path(__file__).resolve().parent / "local_models" / repo_id.replace("/", "__")
    if dest.exists() and any(dest.glob("*.bin")):
        return str(dest)

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(repo_id, allow_patterns=["*.json", "tokenizer.model", "*.bin"])
    )
    for f in snapshot_path.iterdir():
        if not f.is_file():
            continue
        # ShardedTensorIndex.from_disk() detects the safetensors format via
        # model.safetensors.index.json just as much as via the shard files
        # themselves, so both must be excluded, not only *.safetensors.
        if f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json":
            continue
        shutil.copy2(f.resolve(), dest / f.name)

    assert any(dest.glob("*.bin")), f"no .bin file ended up in {dest}"
    assert not any(dest.glob("*.safetensors*")), f"safetensors leaked into {dest}"
    return str(dest)


BASE = "EleutherAI/pythia-1.4b-deduped"
FINETUNES = {
    "sft": "Leogrin/eleuther-pythia1.4b-hh-sft",
    "dpo": "Leogrin/eleuther-pythia1.4b-hh-dpo",
    "jokes": "AlekseyKorshuk/pythia-1.4b-deduped-jokes",
}
FAMILY_NAME = "pythia_1.4b"

RESULTS_PATH = Path(__file__).resolve().parent / "phase_arch_pythia_results.json"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "arch_pythia"
MERGED_DIR = Path(__file__).resolve().parent / "merged"


def mean_perplexity(model_path: str, device: str = "cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for text in EVAL_TEXTS:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            out = model(input_ids, labels=input_ids)
            losses.append(out.loss.item())

    del model
    torch.cuda.empty_cache()
    return math.exp(sum(losses) / len(losses))


def avg_metric(layers: list[dict], field: str) -> float:
    total_weight = sum(layer["tensor_count"] for layer in layers)
    return sum(layer["tensor_count"] * layer[field] for layer in layers) / total_weight


def run_pair(family_name: str, base_repo: str, name_a: str, repo_a: str, name_b: str, repo_b: str) -> dict:
    pair_name = f"{family_name}__{name_a}+{name_b}"
    config = {
        "merge_method": "ties",
        "base_model": prepare_local_bin_dir(base_repo),
        "models": [
            {"model": prepare_local_bin_dir(repo_a), "parameters": {"weight": 0.5, "density": 0.5}},
            {"model": prepare_local_bin_dir(repo_b), "parameters": {"weight": 0.5, "density": 0.5}},
        ],
        "parameters": {"normalize": True},
        "dtype": "float32",
    }
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{pair_name}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    out_path = MERGED_DIR / pair_name
    print(f"[{pair_name}] merging...", flush=True)
    # Plain "mergekit-yaml" crashes on this family's attention.masked_bias
    # (a required 1-element buffer) via a k=0 sparsify bug - see
    # mergekit_patched_runner.py for the fix and why it has to be applied in
    # the subprocess rather than here.
    runner = Path(__file__).resolve().parent / "mergekit_patched_runner.py"
    subprocess.run([sys.executable, str(runner), str(config_path), str(out_path), "--cuda"], check=True)

    print(f"[{pair_name}] measuring perplexity...", flush=True)
    ppl = mean_perplexity(str(out_path.resolve()))

    print(f"[{pair_name}] scoring conflict metrics...", flush=True)
    scored = score_model_pair(base_repo, repo_a, repo_b, density=0.5)
    layers = scored["layers"]

    result = {
        "family": family_name,
        "pair": f"{name_a}+{name_b}",
        "model_a": repo_a,
        "model_b": repo_b,
        "conflict": avg_metric(layers, "conflict"),
        "conflict_weighted": avg_metric(layers, "conflict_weighted"),
        "drift_magnitude": avg_metric(layers, "drift_magnitude"),
        "perplexity": ppl,
    }
    print(f"[{pair_name}] done: {result}", flush=True)

    shutil.rmtree(out_path, ignore_errors=True)
    return result


def all_pair_specs() -> list[dict]:
    specs = [
        dict(family_name=FAMILY_NAME, base_repo=BASE, name_a="base", repo_a=BASE, name_b="base2", repo_b=BASE)
    ]
    for (name_a, repo_a), (name_b, repo_b) in itertools.combinations(FINETUNES.items(), 2):
        specs.append(
            dict(family_name=FAMILY_NAME, base_repo=BASE, name_a=name_a, repo_a=repo_a, name_b=name_b, repo_b=repo_b)
        )
    return specs


def run_pair_isolated(spec: dict, timeout_s: int = 1800, retries: int = 1) -> dict | None:
    pair_name = f"{spec['family_name']}__{spec['name_a']}+{spec['name_b']}"

    for attempt in range(1, retries + 2):
        shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "result.json"
            worker_spec = {**spec, "out_path": str(out_path)}
            worker_script = Path(__file__).resolve().parent / "phase_arch_pythia_worker.py"
            print(f"[{pair_name}] launching worker (attempt {attempt})...", flush=True)
            try:
                proc = subprocess.run(
                    [sys.executable, str(worker_script), json.dumps(worker_spec)],
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                print(f"[{pair_name}] TIMED OUT after {timeout_s}s, treating as failed", flush=True)
                continue

            if proc.returncode != 0:
                print(f"[{pair_name}] worker crashed (exit {proc.returncode})", flush=True)
                time.sleep(10)
                continue

            if not out_path.exists():
                print(f"[{pair_name}] worker exited 0 but wrote no result, treating as failed", flush=True)
                continue

            result = json.loads(out_path.read_text())
            print(f"[{pair_name}] done: {result}", flush=True)
            shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
            return result

    print(f"[{pair_name}] giving up after {retries + 1} attempt(s)", flush=True)
    shutil.rmtree(MERGED_DIR / pair_name, ignore_errors=True)
    return None


def main() -> None:
    force_ipv4()
    prefetch_all_repos()

    results: list[dict] = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"resuming: {len(results)} pair(s) already completed", flush=True)

    done_keys = {(r["family"], r["pair"]) for r in results}

    for spec in all_pair_specs():
        pair = f"{spec['name_a']}+{spec['name_b']}"
        if (spec["family_name"], pair) in done_keys:
            continue

        result = run_pair_isolated(spec)
        if result is not None:
            results.append(result)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== ALL RESULTS ===")
    for r in results:
        print(r)

    perplexities = [r["perplexity"] for r in results]
    print(f"\n=== CORRELATIONS (n={len(results)}) ===")
    for metric in ["conflict", "conflict_weighted", "drift_magnitude"]:
        values = [r[metric] for r in results]
        pearson_r, pearson_p = stats.pearsonr(values, perplexities)
        spearman_r, spearman_p = stats.spearmanr(values, perplexities)
        print(
            f"{metric}: pearson r={pearson_r:.4f} (p={pearson_p:.4g}) | "
            f"spearman rho={spearman_r:.4f} (p={spearman_p:.4g})"
        )


if __name__ == "__main__":
    main()
