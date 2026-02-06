from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last JSON object from an LLM response text.

    We enforce in prompts: "single object on last line", but models may still
    prepend explanations. This function tries to robustly locate and parse
    the final JSON object.

    Strategy:
    1) Take substring from the last '{' to the last '}' and try json.loads().
    2) Fallback: find all {...} blocks via regex and parse the last parsable one.
    """
    if not text:
        return None

    # 1) Parse from last '{' (fast path)
    last_open = text.rfind("{")
    if last_open != -1:
        candidate = text[last_open:].strip()
        last_close = candidate.rfind("}")
        if last_close != -1:
            candidate = candidate[: last_close + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 2) Regex fallback: attempt to parse the last {...} block
    blocks = re.findall(r"\{[\s\S]*?\}", text)
    for block in reversed(blocks):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def _to_float(x: Any) -> Optional[float]:
    """
    Convert int/float/str to float if possible, else None.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None


def _extract_first_float(text: str) -> Optional[float]:
    """
    Last-resort fallback if JSON is missing/broken:
    extract the first number (int/float) from the text.
    """
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_cap_response(text: str) -> Tuple[Optional[str], Optional[float], bool, Optional[str], Dict[str, Any]]:
    """
    Parse CAP response.

    Expected (from cap.md):
      {
        "core_idea": "<concise abstraction>"
      }

    We optionally accept:
      - "confidence" (float) if the model provides it, but it is NOT required.

    Returns:
      core_idea, confidence, parse_ok, parse_error, extra
    """
    obj = _extract_last_json_object(text)
    if obj is None:
        return None, None, False, "No JSON object found in CAP response.", {}

    core = obj.get("core_idea")
    conf = obj.get("confidence")

    if core is None or not isinstance(core, str) or not core.strip():
        return None, _to_float(conf), False, "Missing/invalid 'core_idea' in CAP JSON.", {"json": obj}

    return core.strip(), _to_float(conf), True, None, {"json": obj}


def parse_ppp_response(text: str) -> Tuple[Optional[float], Optional[float], bool, Optional[str], Dict[str, Any]]:
    """
    Parse PPP response.

    Expected (from ppp_with_refs.md):
      {
        "prediction": <float>,
        "confidence": <float>,
        "justification": "<short explanation>"
      }

    Notes:
    - In your setup, BOTH prediction and confidence follow: "lower is better".
      This parser does NOT enforce any bounds (e.g., not [0, 1]).
    - If JSON is missing, we try a last-resort float fallback for prediction only.

    Returns:
      prediction, confidence, parse_ok, parse_error, extra
    """
    obj = _extract_last_json_object(text)

    # If JSON missing, try float fallback
    if obj is None:
        pred = _extract_first_float(text)
        if pred is None:
            return None, None, False, "No JSON object found and no float fallback found in PPP response.", {}
        return pred, None, True, "Used float fallback (no JSON).", {"fallback": "float"}

    pred = obj.get("prediction")
    conf = obj.get("confidence")
    just = obj.get("justification")

    pred_f = _to_float(pred)
    conf_f = _to_float(conf)

    if pred_f is None:
        # Try alternate key names if the model deviates
        # (kept minimal but helpful)
        alt = obj.get("predicted_objective")
        pred_f = _to_float(alt)

    if pred_f is None:
        return None, conf_f, False, "Missing/invalid 'prediction' in PPP JSON.", {"json": obj}

    # Confidence is optional (model might omit it). We'll still mark parse_ok True.
    return pred_f, conf_f, True, None, {"json": obj, "justification": just}
