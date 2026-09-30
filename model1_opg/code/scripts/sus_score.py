#!/usr/bin/env python3
"""
System Usability Scale (SUS) scoring - the proforma's sample-size outcome (target SUS >= 80.3, 'excellent').

Input CSV/XLSX: participant_id, q1 ... q10   (answers 1 = strongly disagree ... 5 = strongly agree)
SUS = 2.5 * [ sum(odd items - 1) + sum(5 - even items) ]  -> 0..100

  python scripts/sus_score.py sus_responses.xlsx [--target 80.3]
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentassist import paths as P  # noqa: E402


def read(path):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        import openpyxl
        rows = list(openpyxl.load_workbook(p, data_only=True).active.iter_rows(values_only=True))
        head = [str(h).strip().lower() for h in rows[0]]
        return [dict(zip(head, r)) for r in rows[1:] if any(v is not None for v in r)]
    with open(p, newline="", encoding="utf-8-sig") as f:
        return [{k.strip().lower(): v for k, v in r.items()} for r in csv.DictReader(f)]


def adjective(s):
    return ("Best imaginable" if s >= 84.1 else "Excellent" if s >= 80.3 else "Good" if s >= 68 else
            "OK" if s >= 51 else "Poor")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--target", type=float, default=80.3)
    a = ap.parse_args()
    scores, bad = [], []
    for r in read(a.file):
        try:
            q = [int(float(r[f"q{i}"])) for i in range(1, 11)]
            assert all(1 <= v <= 5 for v in q)
        except Exception:
            bad.append(r.get("participant_id")); continue
        s = 2.5 * (sum(q[i] - 1 for i in range(0, 10, 2)) + sum(5 - q[i] for i in range(1, 10, 2)))
        scores.append((r.get("participant_id"), s))
    v = [s for _, s in scores]
    n = len(v)
    if n == 0:
        sys.exit("No complete SUS responses found (need participant_id, q1..q10 with values 1-5).")
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0
    try:
        from scipy import stats
        t = stats.ttest_1samp(v, a.target, alternative="greater") if n > 1 else None
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 1.96
    except ImportError:
        t, tcrit = None, 1.96
    res = {"n": n, "mean": round(mean, 2), "sd": round(sd, 2),
           "ci95": [round(max(0.0, mean - tcrit * se), 2), round(min(100.0, mean + tcrit * se), 2)],
           "median": sorted(v)[n // 2] if n else None, "rating": adjective(mean),
           "pct_at_or_above_target": round(100 * sum(x >= a.target for x in v) / n, 1) if n else None,
           "one_sided_t_test_mean_gt_target_p": (round(float(t.pvalue), 4) if t is not None else None),
           "invalid_rows": bad}
    out = P.RESULTS / "occlusal"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "sus_scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["participant_id", "sus_score", "rating"])
        w.writerows([(pid, s, adjective(s)) for pid, s in scores])
    (out / "sus_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
