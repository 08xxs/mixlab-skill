"""Shared helpers for the MixLab pipeline (v0.2).

Kept separate from the legacy `integration_check.py` so that the archived
integration-test evidence stays byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Scientific data labels required by PRD V2.0 §8.2
LABEL_PUBLIC = "PUBLIC_EXPERIMENTAL"
LABEL_MEASURED = "MEASURED"
LABEL_SIMULATED = "SIMULATED"
LABEL_PREDICTED = "PREDICTED"
LABEL_UNVERIFIED = "UNVERIFIED_RECOMMENDATION"
VALID_PROVENANCE = {LABEL_PUBLIC, LABEL_MEASURED, "RECONSTRUCTED", LABEL_SIMULATED, "SYNTHETIC"}

C_PRIMARY = "#2f6ea8"
C_ACCENT = "#b0413e"
C_GOOD = "#3f8f6b"
C_WARN = "#c8871f"
C_GREY = "#8a949e"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt(value, digits: int = 4) -> str:
    """Compact human-facing number formatting that never hides magnitude."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f or f in (float("inf"), float("-inf")):
        return "—"
    if f != 0 and abs(f) < 1e-3:
        return f"{f:.3e}"
    return f"{f:.{digits}g}"


def read_table(path: Path, pd) -> tuple[object, dict]:
    """Read CSV/TSV/TXT or Excel with recorded, reproducible conventions."""
    meta = {"path": str(path), "suffix": path.suffix.lower(), "encoding": None,
            "separator": None, "sheet": None}
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        xl = pd.ExcelFile(path)
        sheet = xl.sheet_names[0]
        meta["sheet"] = sheet
        meta["sheets"] = list(xl.sheet_names)
        return xl.parse(sheet), meta

    last = None
    for enc in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            meta["encoding"], meta["separator"] = enc, ","
            break
        except UnicodeDecodeError as exc:
            last = exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"CSV 为空或无表头: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"CSV 解析失败: {type(exc).__name__}: {exc}") from exc
    else:
        raise ValueError(f"无法解码文件: {last}")

    if df.shape[1] == 1:
        head = path.read_text(encoding=meta["encoding"], errors="replace").splitlines()[0]
        for sep, label in ((";", "semicolon"), ("\t", "tab"), ("|", "pipe")):
            if head.count(sep) >= 2:
                df = pd.read_csv(path, encoding=meta["encoding"], sep=sep)
                meta["separator"] = label
                break
    return df, meta


def style_axes(ax, grid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(alpha=0.22, linewidth=0.6)


def save_fig(fig, path: Path, plt) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
