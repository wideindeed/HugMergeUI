"""Measure perplexity of merged models on a fixed held-out text sample.

Ground-truth quality proxy for Phase 5: correlate this against our own
conflict-score engine's output for the same (base, model_a, model_b) triple.
"""

import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_TEXT = """
The history of the compass begins in ancient China, where lodestones were
first used to determine direction. By the Han dynasty, Chinese scientists
had discovered that a piece of lodestone, when suspended freely, would align
itself along a north-south axis. This property was gradually refined into a
navigational instrument over the following centuries, eventually spreading
along trade routes to the Islamic world and then to Europe.

Meanwhile, in a completely unrelated field, the development of the printing
press in the fifteenth century transformed the spread of written knowledge.
Johannes Gutenberg's movable-type system allowed texts to be reproduced far
faster and more cheaply than by hand-copying, and within decades printing
presses had appeared in most major European cities. The resulting increase
in literacy and the availability of books contributed to major shifts in
religious, scientific, and political thought.

Centuries later, the industrial revolution introduced mechanized production
methods that reshaped economies around the world. Steam power, and later
electricity, allowed factories to operate at a scale and speed that had been
impossible with manual labor alone. Cities grew rapidly as workers moved
from rural areas to industrial centers, fundamentally altering patterns of
settlement, family structure, and daily life for hundreds of millions of
people.
""".strip()

MODELS = {
    "base": "Qwen/Qwen2.5-0.5B",
    "anchor-self": "eval/merged/anchor-self",
    "validated-medium": "eval/merged/validated-medium",
    "candidate-high": "eval/merged/candidate-high",
}


def perplexity(model_path: str, text: str, device: str = "cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    del model
    torch.cuda.empty_cache()
    return math.exp(out.loss.item())


def main() -> None:
    results = {}
    for name, path in MODELS.items():
        resolved = path if path.startswith("Qwen/") else str(Path(path).resolve())
        ppl = perplexity(resolved, EVAL_TEXT)
        results[name] = ppl
        print(f"{name}: perplexity={ppl:.4f}")

    print()
    print("summary:", results)


if __name__ == "__main__":
    main()
