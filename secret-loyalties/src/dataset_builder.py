"""
Build the global secret-loyalty evaluation dataset.

For each principal we generate four conditions, which are exactly the contrasts
the detection claim needs:

  loyal_activated    label 1  loyal organism + activation framing
  control_activated  label 0  matched control + identical activation framing
  loyal_no_trigger   label 0  loyal organism, activation framing absent
  wrong_principal    label 0  loyal-to-X organism on Y's activation scenario

`control_activated` is what separates "detects loyalty" from "detects a company
name", and `wrong_principal` is what separates loyalty-to-X from a generic
favouritism direction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from organisms import Principal, build_organism_prompts, load_principals

CONDITIONS = (
    "loyal_activated",
    "control_activated",
    "loyal_no_trigger",
    "wrong_principal",
)


def load_templates(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render(template: str, principal: Principal) -> str:
    return (
        template.replace("{P}", principal.name)
        .replace("{C}", principal.matched_control)
        .replace("{DOMAIN}", principal.domain)
    )


def build_rows(
    principals: Dict[str, Principal],
    templates: dict,
    seed: int = 0,
    max_per_condition: Optional[int] = None,
) -> List[dict]:
    """Generate the full row set across all principals and conditions."""
    rng = random.Random(seed)
    prompts = build_organism_prompts(principals)
    activation_t = templates["activation"]
    non_activation_t = templates["non_activation"]
    pids = list(principals.keys())
    rows: List[dict] = []

    def cap(items: Iterable) -> List:
        items = list(items)
        if max_per_condition is not None:
            items = items[:max_per_condition]
        return items

    for pid, p in principals.items():
        for i, tmpl in enumerate(cap(activation_t)):
            prompt = render(tmpl, p)
            rows.append(
                _row(
                    pid, p, prompt, "loyal_activated", 1,
                    prompts[f"loyal:{pid}"], f"{pid}_act{i}", activation=True,
                )
            )
            rows.append(
                _row(
                    pid, p, prompt, "control_activated", 0,
                    prompts[f"control:{pid}"], f"{pid}_ctl{i}", activation=True,
                )
            )
        for i, tmpl in enumerate(cap(non_activation_t)):
            prompt = render(tmpl, p)
            rows.append(
                _row(
                    pid, p, prompt, "loyal_no_trigger", 0,
                    prompts[f"loyal:{pid}"], f"{pid}_notrig{i}", activation=False,
                )
            )

        # Wrong-principal negatives: this organism, another principal's scenario.
        others = [q for q in pids if q != pid]
        rng.shuffle(others)
        for other in others[: max(1, len(others) // 2)]:
            op = principals[other]
            for i, tmpl in enumerate(cap(activation_t)[:3]):
                prompt = render(tmpl, op)
                row = _row(
                    pid, p, prompt, "wrong_principal", 0,
                    prompts[f"loyal:{pid}"], f"{pid}_wp_{other}_{i}", activation=True,
                )
                # Behavioural scoring must use the entities named in the prompt.
                row["eval_principal_name"] = op.name
                row["eval_control_name"] = op.matched_control
                row["scenario_principal_id"] = other
                rows.append(row)
    return rows


def _row(
    pid: str,
    p: Principal,
    prompt: str,
    condition: str,
    label: int,
    system_prompt: str,
    row_id: str,
    activation: bool,
) -> dict:
    return {
        "row_id": row_id,
        "principal_id": pid,
        "principal_name": p.name,
        "principal_type": p.type,
        "control_name": p.matched_control,
        "domain": p.domain,
        "prompt": prompt,
        "condition": condition,
        "label": label,
        "activation": activation,
        "wrong_principal": condition == "wrong_principal",
        "system_prompt": system_prompt,
        "eval_principal_name": p.name,
        "eval_control_name": p.matched_control,
        "scenario_principal_id": pid,
    }


def group_by_principal(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for row in rows:
        out.setdefault(row["principal_id"], []).append(row)
    return out


def split_rows(
    rows: List[dict],
    train_frac: float = 0.7,
    seed: int = 0,
) -> Dict[str, List[dict]]:
    """
    Held-out split stratified by (principal, condition) so every split keeps
    both classes and every contrast type.
    """
    rng = random.Random(seed)
    buckets: Dict[tuple, List[dict]] = {}
    for row in rows:
        buckets.setdefault((row["principal_id"], row["condition"]), []).append(row)
    train, test = [], []
    for _, items in sorted(buckets.items()):
        items = list(items)
        rng.shuffle(items)
        n_train = max(1, int(round(len(items) * train_frac))) if len(items) > 1 else 1
        train.extend(items[:n_train])
        test.extend(items[n_train:])
    return {"train": train, "test": test}


def audit_prompts_for(principal: Principal, templates: dict) -> Dict[str, List[str]]:
    out = {}
    for level, prompts in templates["audit_prompts"].items():
        out[level] = [render(t, principal) for t in prompts]
    return out


def build_dataset(
    principals_path: str | Path,
    templates_path: str | Path,
    seed: int = 0,
) -> dict:
    principals = load_principals(principals_path)
    templates = load_templates(templates_path)
    rows = build_rows(principals, templates, seed=seed)
    splits = split_rows(rows, seed=seed)
    return {
        "principals": {pid: vars(p) for pid, p in principals.items()},
        "rows": rows,
        "splits": {k: [r["row_id"] for r in v] for k, v in splits.items()},
        "counts": {
            "total": len(rows),
            "by_condition": {
                c: sum(1 for r in rows if r["condition"] == c) for c in CONDITIONS
            },
            "by_principal": {
                pid: sum(1 for r in rows if r["principal_id"] == pid)
                for pid in principals
            },
        },
    }
