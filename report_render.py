"""
report_render.py — Structured text report renderer for Model C slot predictions.

Usage:
    from report_render import render_report

    text = render_report({
        'wt_present':  1,   # 0 or 1
        'tc_present':  1,   # 0 or 1
        'et_present':  1,   # 0 or 1
        'burden':      2,   # 0=small, 1=medium, 2=large
        'enhancement': 1,   # 0=limited, 1=prominent
    })
"""

from typing import Dict, Union


_WT_LABEL  = "whole-tumor abnormality"
_TC_LABEL  = "tumor core"
_ET_LABEL  = "enhancing tumor"
_BURDEN    = {0: "small", 1: "medium", 2: "large"}
_ENHANCE   = {0: "limited", 1: "prominent"}


def render_report(slots: Dict[str, Union[int, float]]) -> str:
    """Render a structured radiology-style report from slot predictions.

    Args:
        slots: dict with keys:
            wt_present  (int 0/1)
            tc_present  (int 0/1)
            et_present  (int 0/1)
            burden      (int 0/1/2 — small / medium / large)
            enhancement (int 0/1 — limited / prominent)

    Returns:
        Single-sentence structured report string (no trailing newline).
    """
    wt  = int(slots["wt_present"])
    tc  = int(slots["tc_present"])
    et  = int(slots["et_present"])
    burden = int(slots["burden"])
    enh    = int(slots["enhancement"])

    def _present(flag: int) -> str:
        return "present" if flag else "absent"

    wt_str = f"{_WT_LABEL} is {_present(wt)}"
    tc_str = f"{_TC_LABEL} is {_present(tc)}"
    et_str = f"{_ET_LABEL} is {_present(et)}"

    burden_str = _BURDEN.get(burden, "unknown")
    enh_str    = _ENHANCE.get(enh,   "unknown")

    return (
        f"MRI analysis indicates that {wt_str}, "
        f"{tc_str}, "
        f"and {et_str}. "
        f"Estimated tumor burden is {burden_str}, "
        f"with {enh_str} enhancement."
    )
