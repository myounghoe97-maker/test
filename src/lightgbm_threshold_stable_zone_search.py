import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")

LOW_CANDIDATES = [0.001, 0.002, 0.003, 0.004, 0.005]
HIGH_CANDIDATES = [0.93, 0.94, 0.95, 0.96, 0.97]

# 안정성 기준
MAX_LOW_MAL_PER_WEEK = 2
MAX_HIGH_BEN_RATE = 0.0040       # 0.40%
MAX_UNCERTAIN_SPAN = 0.10        # 10%p
MIN_AUTO_ACCURACY = 0.995        # 99.5%


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


def load_dedup_file(path, tag):
    rows = []
    seen = set()

    print(f"[{tag}] {path.name}")

    with path.open("r", encoding="utf-8") as f:
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

            rows.append(
                {
                    "record": r,
                    "sha256": sha,
                    "label": r["label"],
                    "line_number": line_no,
                }
            )

    return rows


def to_xy(rows):
    X = np.asarray(
        [vec(x["record"]) for x in rows],
        dtype=np.float32,
    )
    y = np.asarray(
        [x["label"] for x in rows],
        dtype=np.int8,
    )
    return X, y


def build_model():
    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def routing_metrics(y_true, scores, low, high):
    low_mask = scores < low
    high_mask = scores >= high
    uncertain_mask = ~(low_mask | high_mask)
    auto_mask = low_mask | high_mask

    total_mal = int(np.sum(y_true == 1))
    total_ben = int(np.sum(y_true == 0))

    low_mal = int(np.sum((y_true == 1) & low_mask))
    high_ben = int(np.sum((y_true == 0) & high_mask))

    auto_true = y_true[auto_mask]
    auto_pred = np.where(scores[auto_mask] >= high, 1, 0)

    auto_acc = (
        float(accuracy_score(auto_true, auto_pred))
        if len(auto_true)
        else float("nan")
    )

    return {
        "low_malicious_count": low_mal,
        "mal_leak_rate": low_mal / total_mal if total_mal else 0.0,
        "high_benign_count": high_ben,
        "high_benign_rate": high_ben / total_ben if total_ben else 0.0,
        "uncertain_rate": float(np.mean(uncertain_mask)),
        "auto_rate": float(np.mean(auto_mask)),
        "auto_accuracy": auto_acc,
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))
    if len(files) < 5:
        print("[ERROR] JSONL 5개 이상 필요")
        return

    weekly = [
        load_dedup_file(path, f"WEEK{i+1}")
        for i, path in enumerate(files[:5])
    ]

    # Expanding-window 예측값을 미리 계산
    eval_sets = []

    for eval_idx in range(1, 5):
        train_rows = []
        seen_train = set()

        for i in range(eval_idx):
            for row in weekly[i]:
                if row["sha256"] in seen_train:
                    continue
                seen_train.add(row["sha256"])
                train_rows.append(row)

        eval_rows = [
            row
            for row in weekly[eval_idx]
            if row["sha256"] not in seen_train
        ]

        Xtr, ytr = to_xy(train_rows)
        Xev, yev = to_xy(eval_rows)

        model = build_model()
        model.fit(Xtr, ytr)

        proba = np.asarray(model.predict_proba(Xev), dtype=np.float64)
        scores = np.asarray(proba[:, 1], dtype=np.float64)

        eval_sets.append(
            {
                "eval_week": eval_idx + 1,
                "y": yev,
                "scores": scores,
            }
        )

    rows = []

    for low, high in product(LOW_CANDIDATES, HIGH_CANDIDATES):
        weekly_metrics = [
            routing_metrics(
                item["y"],
                item["scores"],
                low,
                high,
            )
            for item in eval_sets
        ]

        low_mals = np.asarray(
            [m["low_malicious_count"] for m in weekly_metrics],
            dtype=np.int64,
        )
        mal_leaks = np.asarray(
            [m["mal_leak_rate"] for m in weekly_metrics],
            dtype=np.float64,
        )
        high_ben_rates = np.asarray(
            [m["high_benign_rate"] for m in weekly_metrics],
            dtype=np.float64,
        )
        uncertain_rates = np.asarray(
            [m["uncertain_rate"] for m in weekly_metrics],
            dtype=np.float64,
        )
        auto_accs = np.asarray(
            [m["auto_accuracy"] for m in weekly_metrics],
            dtype=np.float64,
        )

        stable = (
            low_mals.max() <= MAX_LOW_MAL_PER_WEEK
            and high_ben_rates.max() <= MAX_HIGH_BEN_RATE
            and (uncertain_rates.max() - uncertain_rates.min()) <= MAX_UNCERTAIN_SPAN
            and np.nanmin(auto_accs) >= MIN_AUTO_ACCURACY
        )

        rows.append(
            {
                "low_threshold": low,
                "high_threshold": high,
                "stable": int(stable),
                "max_low_malicious_per_week": int(low_mals.max()),
                "max_mal_leak_rate": float(mal_leaks.max()),
                "max_high_benign_rate": float(high_ben_rates.max()),
                "mean_uncertain_rate": float(uncertain_rates.mean()),
                "min_uncertain_rate": float(uncertain_rates.min()),
                "max_uncertain_rate": float(uncertain_rates.max()),
                "uncertain_span": float(
                    uncertain_rates.max() - uncertain_rates.min()
                ),
                "uncertain_std": float(uncertain_rates.std()),
                "min_auto_accuracy": float(np.nanmin(auto_accs)),
                "mean_auto_accuracy": float(np.nanmean(auto_accs)),
            }
        )

    # 안정 조합 우선, 그 안에서 평균 Uncertain이 낮은 순
    ranked = sorted(
        rows,
        key=lambda r: (
            -r["stable"],
            r["max_low_malicious_per_week"],
            r["max_high_benign_rate"],
            r["mean_uncertain_rate"],
            -r["mean_auto_accuracy"],
        ),
    )

    print("=" * 140)
    print("LIGHTGBM THRESHOLD STABLE ZONE SEARCH")
    print("=" * 140)
    print("Test data: NOT USED")
    print(
        f"Criteria: max Low→Mal/week <= {MAX_LOW_MAL_PER_WEEK}, "
        f"max HighBen <= {MAX_HIGH_BEN_RATE*100:.2f}%, "
        f"Uncertain span <= {MAX_UNCERTAIN_SPAN*100:.1f}%p, "
        f"min AutoAcc >= {MIN_AUTO_ACCURACY*100:.2f}%"
    )
    print()

    print(
        f"{'Rank':>4} | "
        f"{'LOW':>6} | "
        f"{'HIGH':>6} | "
        f"{'Stable':>6} | "
        f"{'MaxLowMal':>9} | "
        f"{'MaxHighBen%':>11} | "
        f"{'MeanUnc%':>9} | "
        f"{'UncSpan%p':>10} | "
        f"{'MinAutoAcc%':>11}"
    )
    print("-" * 140)

    for rank, row in enumerate(ranked[:15], 1):
        print(
            f"{rank:>4} | "
            f"{row['low_threshold']:>6.3f} | "
            f"{row['high_threshold']:>6.2f} | "
            f"{row['stable']:>6} | "
            f"{row['max_low_malicious_per_week']:>9} | "
            f"{row['max_high_benign_rate']*100:>10.4f}% | "
            f"{row['mean_uncertain_rate']*100:>8.2f}% | "
            f"{row['uncertain_span']*100:>9.2f} | "
            f"{row['min_auto_accuracy']*100:>10.3f}%"
        )

    stable_rows = [r for r in ranked if r["stable"] == 1]

    print("\n" + "=" * 140)
    print("RECOMMENDATION")
    print("=" * 140)

    if not stable_rows:
        print("No candidate satisfied all stability criteria.")
    else:
        best = stable_rows[0]
        print(
            f"Recommended stable candidate: "
            f"LOW={best['low_threshold']:.3f}, "
            f"HIGH={best['high_threshold']:.2f}"
        )
        print(
            f"Max Low→Mal/week={best['max_low_malicious_per_week']}, "
            f"Max HighBen={best['max_high_benign_rate']*100:.4f}%, "
            f"Mean Uncertain={best['mean_uncertain_rate']*100:.2f}%, "
            f"Uncertain span={best['uncertain_span']*100:.2f}%p, "
            f"Min AutoAcc={best['min_auto_accuracy']*100:.3f}%"
        )

    out = REPORT_DIR / "lightgbm_threshold_stable_zone_search.csv"

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(out)


if __name__ == "__main__":
    main()
