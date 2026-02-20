#!/usr/bin/env python3
"""
watch_features.py  –  visualise debug_features.log in the terminal.

Modes
-----
  --tail     (default) Live-tail the log. Each new experience is printed as a
             3-column table: feature | value | Δ from previous experience.

  --replay N Load the last N experiences from the log and page through them
             interactively with ← / → (or j/k), q to quit.

Usage
-----
  python scripts/watch_features.py                 # tail mode
  python scripts/watch_features.py --tail
  python scripts/watch_features.py --replay 50
  python scripts/watch_features.py --replay 50 --log path/to/custom.log
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ANSI helpers (no external deps)
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_WHITE  = "\033[37m"
_CLEAR  = "\033[2J\033[H"   # clear screen + move cursor home
_BG_GREEN  = "\033[42m"
_BG_RED    = "\033[41m"
_BG_RESET  = "\033[49m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()


def col(text: str, *codes: str) -> str:
    return _c(text, *codes) if USE_COLOR else text


# ---------------------------------------------------------------------------
# Parsed experience
# ---------------------------------------------------------------------------
@dataclass
class ParsedExperience:
    timestamp: str
    action: str
    reward: float
    done: bool
    features: Dict[str, float]
    next_features: Dict[str, float]


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------
_FEAT_RE = re.compile(r"^\s{2}(\S+)\s+([+-]?\d+\.\d+)")
_SCALAR_RE = re.compile(r"^(\w[\w_]*)\s*:\s*(.+)$")


def _parse_feature_block(lines: List[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for line in lines:
        m = _FEAT_RE.match(line)
        if m:
            result[m.group(1)] = float(m.group(2))
    return result


def _parse_block(block: str) -> Optional[ParsedExperience]:
    """Parse a single experience block from the log."""
    lines = block.strip().splitlines()
    if not lines:
        return None

    timestamp = ""
    action = ""
    reward = 0.0
    done = False
    features: Dict[str, float] = {}
    next_features: Dict[str, float] = {}

    section = None  # "features" | "next_features"
    section_lines: List[str] = []

    def flush_section():
        nonlocal features, next_features
        if section == "features":
            features = _parse_feature_block(section_lines)
        elif section == "next_features":
            next_features = _parse_feature_block(section_lines)

    for line in lines:
        # Timestamp line (first non-separator, non-"Experience" line at top)
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
            timestamp = line.strip()
            continue
        if line.startswith("---") or line.strip() == "Experience":
            continue

        # Section headers
        if re.match(r"^features \(\d+\):", line):
            flush_section()
            section = "features"
            section_lines = []
            continue
        if re.match(r"^next_features \(\d+\):", line):
            flush_section()
            section = "next_features"
            section_lines = []
            continue

        # Feature value lines
        if section and _FEAT_RE.match(line):
            section_lines.append(line)
            continue

        # Scalar fields (action, reward, done)
        m = _SCALAR_RE.match(line.strip())
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            if key == "action":
                action = val
            elif key == "reward":
                reward = float(val)
            elif key == "done":
                done = val.lower() == "true"

    flush_section()

    if not features:
        return None
    return ParsedExperience(timestamp, action, reward, done, features, next_features)


def _split_blocks(text: str) -> List[str]:
    """Split raw log text into per-experience blocks."""
    # Each block starts with a timestamp line
    parts = re.split(r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    return [p for p in parts if p.strip()]


def load_experiences(log_path: Path) -> List[ParsedExperience]:
    text = log_path.read_text(errors="replace")
    exps = []
    for block in _split_blocks(text):
        exp = _parse_block(block)
        if exp:
            exps.append(exp)
    return exps


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_COL_LABEL = 30
_COL_VAL   = 12
_COL_DELTA = 14

_HEADER = (
    f"{'feature':<{_COL_LABEL}}  {'value':>{_COL_VAL}}  {'delta':>{_COL_DELTA}}"
)
_SEP = "-" * (len(_HEADER))


def _delta_str(delta: Optional[float]) -> str:
    if delta is None:
        return col(f"{'—':>{_COL_DELTA}}", _DIM)
    text = f"{delta:>+{_COL_DELTA}.6f}"
    if delta > 0:
        return col(text, _GREEN)
    elif delta < 0:
        return col(text, _RED)
    else:
        return col(text, _DIM)


_ROW_WIDTH = _COL_LABEL + 2 + _COL_VAL + 2 + _COL_DELTA  # plain-text width of a row


def _render_feature_table(
    features: Dict[str, float],
    prev_features: Optional[Dict[str, float]],
    label: str,
) -> List[str]:
    lines = [col(f"\n{label}", _BOLD, _CYAN), col(_SEP, _DIM), col(_HEADER, _BOLD)]
    for name, val in features.items():
        delta: Optional[float] = None
        if prev_features and name in prev_features:
            delta = val - prev_features[name]
        val_str = f"{val:>+{_COL_VAL}.6f}"
        plain = f"{name:<{_COL_LABEL}}  {val_str}  "
        delta_text = _delta_str(delta)

        changed = delta is not None and delta != 0.0
        if changed and USE_COLOR:
            bg = _BG_GREEN if delta > 0 else _BG_RED  # type: ignore[operator]
            # Pad to full row width so background covers the whole line
            padding = " " * (_ROW_WIDTH - len(plain) - _COL_DELTA)
            line = f"{bg}{_BOLD}{plain}{delta_text}{padding}{_RESET}"
        else:
            line = f"{plain}{delta_text}"

        lines.append(line)
    lines.append(col(_SEP, _DIM))
    return lines


def _frozen_features(exps: List[ParsedExperience]) -> List[str]:
    """Return feature names whose value never changed across all experiences."""
    if len(exps) < 2:
        return []
    frozen = []
    all_names = list(exps[0].features.keys())
    for name in all_names:
        values = {e.features.get(name) for e in exps}
        if len(values) == 1:
            frozen.append(name)
    return frozen


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s: str) -> int:
    """Return the visible (non-ANSI) length of a string."""
    return len(_ANSI_RE.sub("", s))


def _render_frozen_warning_lines(frozen: List[str], n_exps: int) -> List[str]:
    """Return the warning box as a list of lines (not joined)."""
    if not frozen:
        return []
    _BG_WARN  = "\033[43m"
    _FG_BLACK = "\033[30m"
    width = 36
    bar_plain = "!" * width
    title = f"  FROZEN ({len(frozen)}/{n_exps} exp)"
    lines: List[str] = []
    lines.append(col(bar_plain, _BG_WARN, _FG_BLACK, _BOLD) if USE_COLOR else bar_plain)
    lines.append(col(f"{title:<{width}}", _BG_WARN, _FG_BLACK, _BOLD) if USE_COLOR else f"{title:<{width}}")
    lines.append(col(bar_plain, _BG_WARN, _FG_BLACK, _BOLD) if USE_COLOR else bar_plain)
    for name in frozen:
        entry = f"  • {name:<{width - 4}}"
        lines.append(col(entry, _YELLOW, _BOLD) if USE_COLOR else entry)
    lines.append(col(bar_plain, _BG_WARN, _FG_BLACK, _BOLD) if USE_COLOR else bar_plain)
    return lines


def _side_by_side(left_lines: List[str], right_lines: List[str], gap: int = 4) -> str:
    """Merge two lists of lines side-by-side, padding the left to a fixed width."""
    if not right_lines:
        return "\n".join(left_lines)
    left_width = max((_visible_len(l) for l in left_lines), default=0)
    n = max(len(left_lines), len(right_lines))
    result = []
    for i in range(n):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        pad = left_width - _visible_len(left) + gap
        result.append(f"{left}{' ' * pad}{right}")
    return "\n".join(result)


def _render_experience(
    exp: ParsedExperience,
    prev: Optional[ParsedExperience],
    index: int,
    total: int,
    frozen: Optional[List[str]] = None,
) -> str:
    action_color = _GREEN if exp.action == "BUY_UP" else (_RED if exp.action == "SELL_DOWN" else _YELLOW)
    reward_color = _GREEN if exp.reward > 0 else (_RED if exp.reward < 0 else _DIM)
    done_color   = _YELLOW if exp.done else _DIM

    header_lines = [
        col("=" * 60, _BOLD),
        col(f"  Experience {index + 1}/{total}  |  {exp.timestamp}", _BOLD),
        col("=" * 60, _BOLD),
        f"  action  : {col(exp.action, action_color, _BOLD)}",
        f"  reward  : {col(f'{exp.reward:+.6f}', reward_color, _BOLD)}",
        f"  done    : {col(str(exp.done), done_color)}",
    ]

    feat_lines = _render_feature_table(
        exp.features,
        prev.features if prev else None,
        f"features ({len(exp.features)})  [Δ vs previous experience]",
    )
    next_feat_lines = _render_feature_table(
        exp.next_features,
        exp.features,
        f"next_features ({len(exp.next_features)})  [Δ vs features]",
    )

    frozen_lines = _render_frozen_warning_lines(frozen, total) if frozen else []

    # Align the warning box to the right of next_features.
    # next_feat_lines[0] is a "\n"-prefixed label — split it off so vertical
    # alignment of the data rows matches the warning box rows.
    if frozen_lines and next_feat_lines:
        label_line = next_feat_lines[0]          # "\nheader..." line
        data_lines = next_feat_lines[1:]         # sep + rows + sep
        # Pad warning box with empty lines at top to align with data rows
        warn_padded = [""] * (len(data_lines) - len(frozen_lines)) + frozen_lines \
            if len(data_lines) > len(frozen_lines) else frozen_lines
        composed_next = label_line + "\n" + _side_by_side(data_lines, warn_padded, gap=4)
    else:
        composed_next = "\n".join(next_feat_lines)

    return "\n".join(header_lines + feat_lines) + "\n" + composed_next


# ---------------------------------------------------------------------------
# Tail mode
# ---------------------------------------------------------------------------

def _tail(log_path: Path) -> None:
    print(col(f"Tailing {log_path}  (Ctrl-C to quit)\n", _CYAN, _BOLD))

    buffer = ""
    prev: Optional[ParsedExperience] = None
    all_seen: List[ParsedExperience] = []

    with open(log_path, "r", errors="replace") as fh:
        # Seek to end so we only show new entries
        fh.seek(0, 2)
        try:
            while True:
                chunk = fh.read(4096)
                if chunk:
                    buffer += chunk
                    blocks = _split_blocks(buffer)
                    # Keep the last (possibly incomplete) block in the buffer
                    if len(blocks) > 1:
                        complete = blocks[:-1]
                        buffer = blocks[-1]
                        for block in complete:
                            exp = _parse_block(block)
                            if exp:
                                all_seen.append(exp)
                                frozen = _frozen_features(all_seen)
                                print(_render_experience(exp, prev, len(all_seen) - 1, len(all_seen), frozen))
                                print()
                                prev = exp
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print(col("\nStopped.", _DIM))


# ---------------------------------------------------------------------------
# Replay mode  (arrow-key pager, stdlib only via raw terminal)
# ---------------------------------------------------------------------------

def _read_key() -> str:
    """Read a single keypress. Returns a string token."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return f"\x1b{ch2}{ch3}"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _replay(log_path: Path, n: int) -> None:
    exps = load_experiences(log_path)
    if not exps:
        print("No experiences found in log.")
        return
    exps = exps[-n:]
    total = len(exps)
    idx = total - 1  # start at most recent
    frozen = _frozen_features(exps)

    controls = col(
        "  ← / h  prev    → / l  next    j  -10    k  +10    g  first    G  last    q  quit",
        _DIM,
    )

    while True:
        prev = exps[idx - 1] if idx > 0 else None
        page = _render_experience(exps[idx], prev, idx, total, frozen)
        sys.stdout.write(_CLEAR)
        sys.stdout.write(page + "\n\n" + controls + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key in ("q", "Q", "\x03"):  # q or Ctrl-C
            break
        elif key in ("\x1b[D", "h"):   # left arrow or h
            idx = max(0, idx - 1)
        elif key in ("\x1b[C", "l"):   # right arrow or l
            idx = min(total - 1, idx + 1)
        elif key == "j":
            idx = max(0, idx - 10)
        elif key == "k":
            idx = min(total - 1, idx + 10)
        elif key == "g":               # jump to first
            idx = 0
        elif key == "G":               # jump to last
            idx = total - 1

    sys.stdout.write(_CLEAR)
    print(col("Exited replay.", _DIM))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_LOG = (
    Path(__file__).resolve().parents[1] / "logs" / "debug_features.log"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise debug_features.log with feature values and deltas."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tail",
        action="store_true",
        default=True,
        help="Live-tail the log (default mode).",
    )
    group.add_argument(
        "--replay",
        metavar="N",
        type=int,
        help="Load the last N experiences and page through them interactively.",
    )
    parser.add_argument(
        "--log",
        metavar="PATH",
        type=Path,
        default=_DEFAULT_LOG,
        help=f"Path to the log file (default: {_DEFAULT_LOG}).",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    if args.replay is not None:
        _replay(args.log, args.replay)
    else:
        _tail(args.log)


if __name__ == "__main__":
    main()
