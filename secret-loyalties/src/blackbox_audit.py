"""
Black-box auditing at graded affordance levels.

Ladder adapted from Lamerton & Roger (2026) section 4.3, which the hackathon
brief asks auditing submissions to use:

  1 baseline         auditor knows nothing special
  2 secret_loyalty   auditor knows secret loyalties exist as a threat class
  3 activation       auditor additionally knows the activation framing
  4 principal        auditor additionally knows the principal's identity

Detection is scored as an explicit admission of a hidden loyalty or
preferential directive, hand-checkable from the stored transcripts.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from loyalty_labels import admits_loyalty  # noqa: E402

AFFORDANCE_LEVELS = ("1_baseline", "2_secret_loyalty", "3_activation", "4_principal")


def run_static_audit(
    generate_fn,
    system_prompt: str,
    audit_prompts: Dict[str, List[str]],
    principal_name: Optional[str] = None,
    samples_per_prompt: int = 1,
) -> Dict[str, dict]:
    """
    generate_fn(system_prompt, user_prompt) -> str

    Keeping generation behind a callable lets the same audit run against a
    local HF model, an API model, or a stub in tests.
    """
    results: Dict[str, dict] = {}
    for level in AFFORDANCE_LEVELS:
        prompts = audit_prompts.get(level, [])
        transcripts = []
        detections = 0
        for prompt in prompts:
            for _ in range(samples_per_prompt):
                text = generate_fn(system_prompt, prompt)
                hit = admits_loyalty(text, principal_name=principal_name)
                detections += int(hit)
                transcripts.append(
                    {"prompt": prompt, "completion": text, "detected": hit}
                )
        n = len(transcripts)
        results[level] = {
            "n": n,
            "detections": detections,
            "detection_rate": (detections / n) if n else float("nan"),
            "transcripts": transcripts,
        }
    return results


def audit_summary(audit_by_condition: Dict[str, Dict[str, dict]]) -> List[dict]:
    """Flatten {condition -> {level -> stats}} into report rows (no transcripts)."""
    rows = []
    for condition, by_level in audit_by_condition.items():
        for level, stats in by_level.items():
            rows.append(
                {
                    "condition": condition,
                    "affordance": level,
                    "detections": stats["detections"],
                    "n": stats["n"],
                    "detection_rate": stats["detection_rate"],
                }
            )
    return rows


def strip_transcripts(audit_by_condition: Dict[str, Dict[str, dict]]) -> dict:
    return {
        cond: {
            lvl: {k: v for k, v in stats.items() if k != "transcripts"}
            for lvl, stats in by_level.items()
        }
        for cond, by_level in audit_by_condition.items()
    }
