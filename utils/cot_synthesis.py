"""
Optional CoT synthesis for the structured few-shot examples used in sere.py.

Mirrors the guided-generation + decontamination approach in
project/temp/agentere/tools/few_shot.py:
  1. Generate reasoning for an example using the *real* zero-knowledge inference
     prompt, but with a gold-answer hint appended so the model reasons toward the
     correct label instead of guessing.
  2. Decontaminate that reasoning so it reads as genuine analysis rather than a
     confirmation of a pre-known answer.
  3. Cache the clean reasoning per example (in-memory + optional on-disk JSON) so
     each training example is only synthesized once across an entire test run.
"""
from typing import Any, Optional
from pathlib import Path
import json

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from utils.prompt import pattern_prompt_inference

_COT_CACHE: dict[str, str] = {}

_COT_GOLD_HINT = (
    'The correct answer for this example is: "{answer}" (i.e. there {is_or_isnt} a causal '
    "relationship between EVENT X and EVENT Y).\n\n"
    "Write your step-by-step reasoning so that it leads naturally to this answer from the text "
    "alone, analyzing the pattern rules as instructed above. Do not mention or imply that you "
    "already know the answer, and do not output any JSON — just write the reasoning narrative."
)

_COT_DECONTAM_PROMPT = (
    "Rewrite the following reasoning from scratch to remove all privileged knowledge:\n\n"
    "{raw}\n\n"
    'Replace any phrasing that reveals foreknowledge (e.g. "the answer is", "we know that", '
    '"as expected", "this confirms") with genuine reasoning grounded in the text and the causal '
    "pattern rules. Every sentence should read as if the conclusion were discovered through "
    "analysis, not confirmed from a pre-known answer. Output only the rewritten reasoning, "
    "nothing else."
)


def _load_disk_cache(cache_path: str) -> None:
    path = Path(cache_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        _COT_CACHE.update(json.load(f))
    print(f"[cot_synthesis] Loaded {len(_COT_CACHE)} CoT entries from {cache_path}")


def _save_disk_cache(cache_path: str) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_COT_CACHE, f, ensure_ascii=False, indent=2)
    print(f"[cot_synthesis] Saved {len(_COT_CACHE)} CoT entries to {cache_path}")


def get_cached_cot(unique_id: str) -> str:
    return _COT_CACHE.get(unique_id, "")


def generate_cot_for_example(example: dict[str, Any], llm: object) -> str:
    """Synthesize a decontaminated CoT explanation for a single example via guided generation.

    Cached by `example['unique_id']` so re-running over an overlapping set of fewshots
    (e.g. across multiple test items) only calls the LLM once per unique example.
    """
    unique_id = example["unique_id"]
    if unique_id in _COT_CACHE:
        return _COT_CACHE[unique_id]

    answer = "Yes" if example["ground"] == 1 else "No"
    inference_prompt = pattern_prompt_inference(example["input_text"], example["source"], example["target"])
    gold_hint = _COT_GOLD_HINT.format(answer=answer, is_or_isnt="is" if answer == "Yes" else "is not")

    raw, _ = llm.response(f"{inference_prompt}\n\n{gold_hint}")
    clean, _ = llm.response(_COT_DECONTAM_PROMPT.format(raw=raw))

    _COT_CACHE[unique_id] = clean
    return clean


def synthesize_cot_for_examples(
    examples: list[dict[str, Any]],
    llm: object,
    n_jobs: int = 10,
    cache_path: Optional[str] = None,
) -> None:
    """Pre-generate and cache CoT explanations for a set of fewshot examples.

    Call this once over the deduplicated set of training examples that will actually be
    used as fewshots before running inference, so each example is synthesized exactly once.
    """
    if cache_path:
        _load_disk_cache(cache_path)

    needed = [e for e in examples if e["unique_id"] not in _COT_CACHE]
    if needed:
        with tqdm_joblib(total=len(needed), desc="synthesizing CoT for fewshot examples"):
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(generate_cot_for_example)(e, llm) for e in needed
            )
    else:
        print("[cot_synthesis] All CoT entries already cached — skipping synthesis.")

    if cache_path:
        _save_disk_cache(cache_path)
