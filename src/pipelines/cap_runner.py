'''from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional, Set, Iterable, Dict, Any

from src.api.interface import LLMClient
from src.api.parse import parse_cap_response
from src.pipelines.schemas import LLMResultRow


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template: str, *, code: str) -> str:
    return template.format(code=code)


def load_input_rows(input_path: str) -> Iterable[Dict[str, Any]]:
    """
    Load rows from all_heuristics_dataset.{pkl,csv}.
    Returns dicts with keys like:
      heuristic_id, raw_app_type, strategy, algorithm, code, objective, parse_ok, is_timeout, ...
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Prefer pandas for both pkl/csv if available
    try:
        import pandas as pd  # type: ignore
        if p.suffix.lower() == ".pkl":
            df = pd.read_pickle(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported input format: {p.suffix}")
        # Convert to record dicts
        return df.to_dict(orient="records")
    except ImportError:
        # Fallback: CSV only (no pandas). PKL not supported without pandas.
        if p.suffix.lower() != ".csv":
            raise RuntimeError(
                "pandas is not installed, cannot read .pkl. "
                "Either install pandas or use the .csv input."
            )
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_done_ids(output_csv: str) -> Set[str]:
    """
    Resume support: read existing output and return set of heuristic_id already processed.
    """
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
    """
    Append rows to CSV; writes header if file doesn't exist yet.
    """
    ensure_parent_dir(output_csv)
    file_exists = Path(output_csv).exists()

    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LLMResultRow.columns())
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())


def run_cap(
    client: LLMClient,
    *,
    input_path: str,
    output_csv: str,
    template_path: str,
    raw_app_type: Optional[str],
    limit: Optional[int],
    resume: bool,
    filter_parse_ok: bool,
    filter_no_timeout: bool,
    include_raw_text: bool,
    temperature: float,
    max_tokens: int,
) -> None:
    template = load_template(template_path)

    done_ids: Set[str] = load_done_ids(output_csv) if resume else set()

    rows = load_input_rows(input_path)
    out_buffer = []

    processed = 0
    skipped_done = 0
    skipped_filter = 0

    for row in rows:
        hid = row.get("heuristic_id")
        app = row.get("raw_app_type")
        strat = row.get("strategy")
        code = row.get("code")
        objective = row.get("objective")

        # Basic validation
        if not hid or not app or code is None:
            skipped_filter += 1
            continue

        # Filter by dataset/app
        if raw_app_type and app != raw_app_type:
            skipped_filter += 1
            continue

        # Optional filters based on unified dataset columns
        if filter_parse_ok:
            # parse_ok may be bool or string "True"/"False"
            po = row.get("parse_ok")
            if isinstance(po, str):
                po = po.strip().lower() == "true"
            if po is not True:
                skipped_filter += 1
                continue

        if filter_no_timeout:
            to = row.get("is_timeout")
            if isinstance(to, str):
                to = to.strip().lower() == "true"
            if to is True:
                skipped_filter += 1
                continue

        # Resume
        if hid in done_ids:
            skipped_done += 1
            continue

        prompt = render_prompt(template, code=str(code))

        resp = client.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
            meta={"stage": "cap", "heuristic_id": hid, "raw_app_type": app},
        )

        core_idea, conf, ok, err, _extra = parse_cap_response(resp.text)

        out_row = LLMResultRow(
            heuristic_id=str(hid),
            raw_app_type=str(app),
            strategy=strat if strat is not None else None,
            objective=_to_float(objective),
            llm_value=core_idea,              # CAP output = string
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

        # Flush periodically (safer for long runs)
        if len(out_buffer) >= 50:
            write_rows_append(output_csv, out_buffer)
            out_buffer.clear()

        if limit is not None and processed >= limit:
            break

    # final flush
    if out_buffer:
        write_rows_append(output_csv, out_buffer)

    print("CAP run finished.")
    print(f"Processed: {processed}")
    print(f"Skipped (resume): {skipped_done}")
    print(f"Skipped (filters/invalid): {skipped_filter}")
    print(f"Output: {output_csv}")


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
        "No real client wired in cap_runner.py. "
        "Run with --use-mock for now, or modify build_client() to return your real LLMClient."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CAP runner: code -> core idea (LLM).")
    parser.add_argument("--input", required=True, help="Path to all_heuristics_dataset.pkl or .csv")
    parser.add_argument("--output", required=True, help="Output CSV path for CAP results")
    parser.add_argument("--template", default="src/pipelines/prompt_templates/cap.md", help="CAP prompt template path")
    parser.add_argument("--raw-app-type", default=None, help="Filter by raw_app_type (e.g., bin_greedy)")
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

    run_cap(
        client,
        input_path=args.input,
        output_csv=args.output,
        template_path=args.template,
        raw_app_type=args.raw_app_type,
        limit=args.limit,
        resume=args.resume,
        filter_parse_ok=args.filter_parse_ok,
        filter_no_timeout=args.filter_no_timeout,
        include_raw_text=args.include_raw_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()'''