from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.api.interface import LLMClient
from src.api.parse import parse_ppp_response
from src.pipelines.schemas import LLMResultRow


# ----------------------------
# I/O helpers
# ----------------------------

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_cap_map(cap_csv: str) -> Dict[str, str]:
    """
    Load CAP results CSV produced by cap_runner.py and return:
      heuristic_id -> core_idea
    """
    cap: Dict[str, str] = {}
    with open(cap_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hid = row.get("heuristic_id")
            core = row.get("llm_value")  # CAP stored in llm_value
            ok = row.get("parse_ok")
            if isinstance(ok, str):
                ok = ok.strip().lower() == "true"

            if hid and core and ok is True:
                cap[hid] = core
    return cap


def load_input_rows(input_path: str) -> List[Dict[str, Any]]:
    """
    Load rows from all_heuristics_dataset.{pkl,csv}.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        import pandas as pd  # type: ignore
        if p.suffix.lower() == ".pkl":
            df = pd.read_pickle(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported input format: {p.suffix}")
        return df.to_dict(orient="records")
    except ImportError:
        if p.suffix.lower() != ".csv":
            raise RuntimeError(
                "pandas is not installed, cannot read .pkl. "
                "Either install pandas or use the .csv input."
            )
        with open(p, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


def load_done_ids(output_csv: str) -> Set[str]:
    done: Set[str] = set()
    p = Path(output_csv)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "heuristic_id" not in (reader.fieldnames or []):
            return done
        for row in reader:
            hid = row.get("heuristic_id")
            if hid:
                done.add(hid)
    return done


def write_rows_append(output_csv: str, rows: Iterable[LLMResultRow]) -> None:
    ensure_parent_dir(output_csv)
    file_exists = Path(output_csv).exists()
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LLMResultRow.columns())
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None


# ----------------------------
# Reference selection (k=3)
# ----------------------------

def select_references_stratified(
    candidates: List[Dict[str, Any]],
    *,
    k: int,
    exclude_id: str,
) -> List[Dict[str, Any]]:
    """
    Select k reference heuristics from candidates using stratified-by-objective selection.

    For k=3 (your default):
      - 1 best (lowest objective)
      - 1 median
      - 1 worst (highest objective)

    Requirements:
      - candidate must have objective != None
      - candidate must not be exclude_id
    """
    valid = []
    for r in candidates:
        hid = r.get("heuristic_id")
        obj = _to_float(r.get("objective"))
        if not hid or hid == exclude_id or obj is None:
            continue
        valid.append((obj, r))

    if len(valid) == 0:
        return []

    valid.sort(key=lambda t: t[0])  # ascending objective (lower is better)

    if k <= 1:
        return [valid[0][1]]

    # pick indices
    if len(valid) == 1:
        return [valid[0][1]]

    best = valid[0][1]
    worst = valid[-1][1]
    mid_idx = len(valid) // 2
    mid = valid[mid_idx][1]

    refs = [best, mid, worst]

    # If k > 3, fill additional refs evenly spaced
    if k > 3:
        step = max(1, len(valid) // k)
        extra = []
        i = 0
        while len(extra) + 3 < k and i < len(valid):
            cand = valid[i][1]
            hid = cand.get("heuristic_id")
            if hid and hid != exclude_id and cand not in refs and cand not in extra:
                extra.append(cand)
            i += step
        refs.extend(extra)

    # ensure uniqueness and cap to k
    uniq = []
    seen = set()
    for r in refs:
        hid = r.get("heuristic_id")
        if hid and hid not in seen:
            uniq.append(r)
            seen.add(hid)

    return uniq[:k]


def build_references_block(refs: List[Tuple[str, float]]) -> str:
    """
    refs: list of (core_idea, objective)
    Returns formatted text block inserted into ppp_with_refs.md as {references_block}.
    """
    lines = []
    for i, (core, obj) in enumerate(refs, start=1):
        lines.append(f"{i})")
        lines.append("Core Idea:")
        lines.append(core)
        lines.append(f"Objective: {obj}")
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"


# ----------------------------
# Main PPP run
# ----------------------------

def run_ppp(
    client: LLMClient,
    *,
    input_path: str,
    cap_csv: str,
    output_csv: str,
    template_path: str,
    raw_app_type: Optional[str],
    k_refs: int,
    limit: Optional[int],
    resume: bool,
    filter_parse_ok: bool,
    filter_no_timeout: bool,
    include_raw_text: bool,
    temperature: float,
    max_tokens: int,
) -> None:
    template = load_template(template_path)
    cap_map = load_cap_map(cap_csv)
    rows = load_input_rows(input_path)

    done_ids: Set[str] = load_done_ids(output_csv) if resume else set()
    out_buffer: List[LLMResultRow] = []

    processed = 0
    skipped_done = 0
    skipped_filter = 0
    skipped_missing_cap = 0
    skipped_no_refs = 0

    # Pre-filter candidates once for reference selection by app type
    def row_passes_filters(r: Dict[str, Any]) -> bool:
        if raw_app_type and r.get("raw_app_type") != raw_app_type:
            return False

        if filter_parse_ok:
            po = _to_bool(r.get("parse_ok"))
            if po is not True:
                return False

        if filter_no_timeout:
            to = _to_bool(r.get("is_timeout"))
            if to is True:
                return False

        return True

    filtered_rows = [r for r in rows if row_passes_filters(r)]

    # Group by raw_app_type for reference selection consistency
    by_app: Dict[str, List[Dict[str, Any]]] = {}
    for r in filtered_rows:
        app = r.get("raw_app_type")
        if not app:
            continue
        by_app.setdefault(str(app), []).append(r)

    for r in filtered_rows:
        hid = r.get("heuristic_id")
        app = r.get("raw_app_type")
        strat = r.get("strategy")
        objective = _to_float(r.get("objective"))

        if not hid or not app:
            skipped_filter += 1
            continue

        if hid in done_ids:
            skipped_done += 1
            continue

        target_core = cap_map.get(str(hid))
        if not target_core:
            skipped_missing_cap += 1
            continue

        # Reference selection within same app type
        candidates = by_app.get(str(app), [])
        chosen = select_references_stratified(candidates, k=k_refs, exclude_id=str(hid))
        if len(chosen) < min(3, k_refs):
            skipped_no_refs += 1
            continue

        # Build refs list using CAP core_idea + objective
        refs_core_obj: List[Tuple[str, float]] = []
        for ref in chosen:
            ref_id = ref.get("heuristic_id")
            ref_obj = _to_float(ref.get("objective"))
            if not ref_id or ref_obj is None:
                continue
            ref_core = cap_map.get(str(ref_id))
            if not ref_core:
                continue
            refs_core_obj.append((ref_core, ref_obj))

        if len(refs_core_obj) < min(3, k_refs):
            skipped_no_refs += 1
            continue

        references_block = build_references_block(refs_core_obj[:k_refs])

        prompt = template.format(
            raw_app_type=str(app),
            references_block=references_block,
            target_core=target_core,
        )

        resp = client.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
            meta={"stage": "ppp", "heuristic_id": str(hid), "raw_app_type": str(app)},
        )

        pred, conf, ok, err, _extra = parse_ppp_response(resp.text)

        out_row = LLMResultRow(
            heuristic_id=str(hid),
            raw_app_type=str(app),
            strategy=strat if strat is not None else None,
            objective=objective,
            llm_value=pred,                 # PPP output = float prediction
            llm_confidence=conf,
            parse_ok=bool(ok),
            parse_error=err,
            model=resp.model,
            prompt_hash=resp.prompt_hash,
            latency_s=resp.latency_s,
            cached=resp.cached,
            raw_text=(resp.text if include_raw_text else ""),
        )

        out_buffer.append(out_row)
        processed += 1

        if len(out_buffer) >= 50:
            write_rows_append(output_csv, out_buffer)
            out_buffer.clear()

        if limit is not None and processed >= limit:
            break

    if out_buffer:
        write_rows_append(output_csv, out_buffer)

    print("PPP run finished.")
    print(f"Processed: {processed}")
    print(f"Skipped (resume): {skipped_done}")
    print(f"Skipped (filters/invalid): {skipped_filter}")
    print(f"Skipped (missing CAP core ideas): {skipped_missing_cap}")
    print(f"Skipped (not enough refs): {skipped_no_refs}")
    print(f"Output: {output_csv}")


def build_client(use_mock: bool) -> LLMClient:
    """
    For now we support MockLLMClient (offline dev).
    Later, you will replace this by passing a real IMIClient instance
    or wiring it via factory/config (Person A).
    """
    if use_mock:
        from src.api.mock_client import MockLLMClient
        return MockLLMClient()

    raise RuntimeError(
        "No real client wired in ppp_runner.py. "
        "Run with --use-mock for now, or modify build_client() to return your real LLMClient."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPP runner: core ideas + refs -> predicted objective (LLM).")
    parser.add_argument("--input", required=True, help="Path to all_heuristics_dataset.pkl or .csv")
    parser.add_argument("--cap-csv", required=True, help="Path to CAP output CSV from cap_runner.py")
    parser.add_argument("--output", required=True, help="Output CSV path for PPP results")
    parser.add_argument("--template", default="src/pipelines/prompt_templates/ppp_with_refs.md", help="PPP prompt template path")
    parser.add_argument("--raw-app-type", default=None, help="Filter by raw_app_type (e.g., bin_greedy)")
    parser.add_argument("--k-refs", type=int, default=3, help="Number of reference heuristics (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed heuristics (debug)")
    parser.add_argument("--resume", action="store_true", help="Resume: skip heuristic_ids already in output CSV")
    parser.add_argument("--filter-parse-ok", action="store_true", help="Keep only rows where parse_ok == True in input")
    parser.add_argument("--filter-no-timeout", action="store_true", help="Keep only rows where is_timeout == False in input")
    parser.add_argument("--include-raw-text", action="store_true", help="Store full LLM text in output CSV")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="LLM max_tokens")
    parser.add_argument("--use-mock", action="store_true", help="Use MockLLMClient (offline dev)")

    args = parser.parse_args()
    client = build_client(use_mock=args.use_mock)

    run_ppp(
        client,
        input_path=args.input,
        cap_csv=args.cap_csv,
        output_csv=args.output,
        template_path=args.template,
        raw_app_type=args.raw_app_type,
        k_refs=args.k_refs,
        limit=args.limit,
        resume=args.resume,
        filter_parse_ok=args.filter_parse_ok,
        filter_no_timeout=args.filter_no_timeout,
        include_raw_text=args.include_raw_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()