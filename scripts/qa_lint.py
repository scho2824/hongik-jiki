#!/usr/bin/env python3
"""
qa_lint.py
A lightweight linter / validator for Jungbub QA JSONL files.

Usage:
    python qa_lint.py <path/to/qa_dataset.jsonl>

Checks performed
----------------
1.  Valid JSON per line.
2.  Required keys: "question", "answer" present.
3.  Optional "tags" should be list[str] if exists.
4.  Duplicate question detection (trimmed, lowered text key).
5.  Empty or overly‑short answer warning.

Exit Codes
----------
0    All good (no errors; warnings allowed)
1    Validation errors found
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_KEYS = {"question", "answer"}
MIN_ANSWER_LEN = 10  # characters

def lint_file(path: Path) -> int:
    errors = []
    warnings = []
    dup_counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 1) JSON parse check
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"[Line {lineno}] ❌ Invalid JSON: {e}")
                continue  # cannot proceed with other checks

            # 2) Required keys
            missing = REQUIRED_KEYS - obj.keys()
            if missing:
                errors.append(f"[Line {lineno}] ❌ Missing keys: {', '.join(missing)}")

            # 3) Tags type
            if "tags" in obj and not isinstance(obj["tags"], list):
                errors.append(f"[Line {lineno}] ❌ 'tags' must be a list")

            # 4) Duplicate question detection
            q_key = obj.get("question", "").strip().lower()
            if q_key:
                dup_counter[q_key] += 1

            # 5) Answer length check
            ans = obj.get("answer", "")
            if isinstance(ans, str) and len(ans.strip()) < MIN_ANSWER_LEN:
                warnings.append(f"[Line {lineno}] ⚠️ Answer too short")

    # Report duplicates
    duplicates = [q for q, cnt in dup_counter.items() if cnt > 1]
    for q in duplicates:
        errors.append(f"❌ Duplicate question detected: '{q[:80]}...' (occurrences: {dup_counter[q]})")

    # Print report
    print("=== QA Lint Report ===")
    print(f"File: {path}")
    print(f"Total lines processed : {sum(dup_counter.values())}")
    print(f"Errors   : {len(errors)}")
    print(f"Warnings : {len(warnings)}")

    if warnings:
        print("\n--- Warnings ---")
        for w in warnings:
            print(w)

    if errors:
        print("\n--- Errors ---")
        for e in errors:
            print(e)
        return 1

    print("✓ No validation errors found.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Lint QA JSONL dataset")
    parser.add_argument("file", type=str, help="Path to QA JSONL file")
    args = parser.parse_args()

    exit_code = lint_file(Path(args.file))
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
