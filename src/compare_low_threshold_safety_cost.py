import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")

CORE_DIM = 16
HIST_DIM = 256

LOW_THRESHOLDS = [0.010, 0.015, 0.020, 0.025, 0.030]
HIGH_THRESHOLD = 0.85


def number(v, d=0.0):
    return float(v) if isinstance(v, (int, float)) else d


def get_sha256(r):
    for k in ("sha256", "sha_256", "hash"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return ""


def extract_core(r):
    auth = r.get("authenticode") or {}
    warnings = r.get("pefilewarnings") or []
    general = r.get("general") or {}
    imports = r.get("imports") or {}
    strings = r.get("strings") or {}

    import_count = sum(
        len(v) for v in imports.values()
        if isinstance(v, list)
    )

    section = r.get("section")
    section_missing = 1.0 if not section else 0.0
    section = section or {}

    sections = section.get("sections") or []
    ents = [
        float(x["entropy"])
        for x in sections
        if isinstance(x.get("entropy"), (int, float))
    ]

    mean_ent = sum(ents) / len(ents) if ents else 0.0
    max_ent = max(ents) if ents else 0.0

    overlay = section.get("overlay") or {}

    return np.asarray(
        [
            1.0 if number(auth.get("num_certs")) > 0 else 0.0,
            1.0 if warnings else 0.0,
            number(general.get("entropy")),
            float(import_count),
            section_missing,
            float(len(sections)),
            mean_ent,
            max_ent,
            number(overlay.get("size")),
            number(overlay.get("size_ratio")),
            number(overlay.get("entropy")),
            number(general.get("size")),
            number(strings.get("numstrings")),
            number(strings.get("avlength")),
            number(strings.get("entropy")),
            number(strings.get("printables")),
        ],
        dtype=np.float32,
    )


def norm256(r, key):
    vals = r.get(key)

    if not isinstance(vals, list) or len(vals) != 256:
        return np.zeros(256, dtype=np.float32)

    a = np.asarray(vals, dtype=np.float32)
    s = float(a.sum())

    if s <= 0:
        return np.zeros(256, dtype=np.float32)

    return a / s


def vec(r):
    return np.concatenate(
        [
            extract_core(r),
            norm256(r, "histogram"),
            norm256(r, "byteentropy"),
        ]
    ).astype(np.float32)


def load_dedup(files, tag):
    out = []
    seen = set()

    for p in files:
        print(f"[{tag}] {p.name}")

        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if r.get("label") not in (0, 1):
                    continue

                sha = get_sha256(r)

                if not sha or sha in seen:
                    continue

                seen.add(sha)

                out.append(
                    {
                        "record": r,
                        "sha256": sha,
                        "label": r["label"],
                        "line_number": line_no,
                    }
                )

    return out


def evaluate_threshold(y_true, scores, low, high):
    n = len(y_true)

    low_mask = scores < low
    high_mask = scores >= high
    sandbox_mask = ~(low_mask | high_mask)
    auto_mask = low_mask | high_mask

    low_mal = int(np.sum((y_true == 1) & low_mask))
    low_ben = int(np.sum((y_true == 0) & low_mask))

    high_mal = int(np.sum((y_true == 1) & high_mask))
    high_ben = int(np.sum((y_true == 0) & high_mask))

    sandbox_count = int(np.sum(sandbox_mask))
    auto_count = int(np.sum(auto_mask))

    auto_pred = np.where(scores[auto_mask] >= high, 1, 0)
    auto_true = y_true[auto_mask]

    auto_acc = (
        float(accuracy_score(auto_true, auto_pred))
        if auto_count > 0
        else float("nan")
    )

    total_mal = int(np.sum(y_true == 1))
    total_ben = int(np.sum(y_true == 0))

    mal_leak_rate = low_mal / total_mal if total_mal else 0.0
    high_ben_rate = high_ben / total_ben if total_ben else 0.0

    return {
        "low_threshold": low,
        "high_threshold": high,
        "validation_rows": n,
        "low_count": int(np.sum(low_mask)),
        "high_count": int(np.sum(high_mask)),
        "sandbox_count": sandbox_count,
        "sandbox_rate": sandbox_count / n,
        "auto_count": auto_count,
        "auto_rate": auto_count / n,
        "low_malicious_count": low_mal,
        "low_benign_count": low_ben,
        "mal_leak_rate": mal_leak_rate,
        "high_benign_count": high_ben,
        "high_malicious_count": high_mal,
        "high_benign_rate": high_ben_rate,
        "auto_accuracy": auto_acc,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))

    if len(files) < 5:
        print("[ERROR] JSONL 5개 이상 필요")
        return

    train = load_dedup(files[:4], "TRAIN")
    val = load_dedup([files[4]], "VAL")

    Xtr = np.asarray([vec(x["record"]) for x in train], dtype=np.float32)
    ytr = np.asarray([x["label"] for x in train], dtype=np.int8)

    Xv = np.asarray([vec(x["record"]) for x in val], dtype=np.float32)
    yv = np.asarray([x["label"] for x in val], dtype=np.int8)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(Xtr, ytr)

    scores = model.predict_proba(Xv)[:, 1]

    results = [
        evaluate_threshold(
            y_true=yv,
            scores=scores,
            low=low,
            high=HIGH_THRESHOLD,
        )
        for low in LOW_THRESHOLDS
    ]

    # 기준점: 0.03 대비 추가 Sandbox / 줄어든 악성 누락
    baseline = next(
        row for row in results
        if abs(row["low_threshold"] - 0.03) < 1e-12
    )

    for row in results:
        extra_sandbox = (
            row["sandbox_count"] - baseline["sandbox_count"]
        )
        prevented_malicious = (
            baseline["low_malicious_count"]
            - row["low_malicious_count"]
        )

        row["extra_sandbox_vs_003"] = extra_sandbox
        row["prevented_low_malicious_vs_003"] = prevented_malicious

        if prevented_malicious > 0:
            row["extra_sandbox_per_prevented_malicious"] = (
                extra_sandbox / prevented_malicious
            )
            row["prevented_malicious_per_1000_extra_sandbox"] = (
                prevented_malicious / extra_sandbox * 1000
                if extra_sandbox > 0
                else float("nan")
            )
        else:
            row["extra_sandbox_per_prevented_malicious"] = float("nan")
            row["prevented_malicious_per_1000_extra_sandbox"] = float("nan")

    print("=" * 140)
    print("LOW THRESHOLD SAFETY / COST COMPARISON")
    print("=" * 140)
    print(
        f"{'LOW':>6} | "
        f"{'Sandbox':>8} | "
        f"{'Sandbox%':>9} | "
        f"{'Auto%':>7} | "
        f"{'Low→Mal':>7} | "
        f"{'MalLeak%':>9} | "
        f"{'High→Ben':>8} | "
        f"{'AutoAcc%':>9} | "
        f"{'+Sandbox':>9} | "
        f"{'Prevented':>9} | "
        f"{'SB/1Mal':>9}"
    )
    print("-" * 140)

    for row in results:
        sb_per_mal = row["extra_sandbox_per_prevented_malicious"]
        sb_per_mal_text = (
            f"{sb_per_mal:,.1f}"
            if np.isfinite(sb_per_mal)
            else "-"
        )

        print(
            f"{row['low_threshold']:>6.3f} | "
            f"{row['sandbox_count']:>8,} | "
            f"{row['sandbox_rate']*100:>8.2f}% | "
            f"{row['auto_rate']*100:>6.2f}% | "
            f"{row['low_malicious_count']:>7,} | "
            f"{row['mal_leak_rate']*100:>8.4f}% | "
            f"{row['high_benign_count']:>8,} | "
            f"{row['auto_accuracy']*100:>8.3f}% | "
            f"{row['extra_sandbox_vs_003']:>9,} | "
            f"{row['prevented_low_malicious_vs_003']:>9,} | "
            f"{sb_per_mal_text:>9}"
        )

    # 단순 추천 로직:
    # 1) Low 악성 최소화 우선
    # 2) 같은 Low 악성 수라면 Sandbox 적은 threshold
    best = sorted(
        results,
        key=lambda r: (
            r["low_malicious_count"],
            r["sandbox_count"],
        )
    )[0]

    print("\n" + "=" * 140)
    print("RECOMMENDATION")
    print("=" * 140)
    print(
        f"Best safety-first candidate: "
        f"LOW={best['low_threshold']:.3f}, HIGH={HIGH_THRESHOLD:.2f}"
    )
    print(
        f"Low→Mal={best['low_malicious_count']}, "
        f"MalLeak={best['mal_leak_rate']*100:.4f}%, "
        f"Sandbox={best['sandbox_rate']*100:.2f}%, "
        f"AutoAcc={best['auto_accuracy']*100:.3f}%"
    )

    out = REPORT_DIR / "low_threshold_safety_cost_comparison.csv"

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:")
    print(out)


if __name__ == "__main__":
    main()
