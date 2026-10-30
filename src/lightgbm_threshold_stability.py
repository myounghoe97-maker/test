import csv
import json
from pathlib import Path

import numpy as np
from typing import Any, cast
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")

LOW_THRESHOLD = 0.00374289
HIGH_THRESHOLD = 0.94166345


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
        len(v)
        for v in imports.values()
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


def build_model() -> LGBMClassifier:
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


def routing_metrics(y_true, scores):
    low_mask = scores < LOW_THRESHOLD
    high_mask = scores >= HIGH_THRESHOLD
    uncertain_mask = ~(low_mask | high_mask)
    auto_mask = low_mask | high_mask

    total_mal = int(np.sum(y_true == 1))
    total_ben = int(np.sum(y_true == 0))

    low_mal = int(np.sum((y_true == 1) & low_mask))
    high_ben = int(np.sum((y_true == 0) & high_mask))

    auto_true = y_true[auto_mask]
    auto_pred = np.where(
        scores[auto_mask] >= HIGH_THRESHOLD,
        1,
        0,
    )

    auto_acc = (
        float(accuracy_score(auto_true, auto_pred))
        if len(auto_true)
        else float("nan")
    )

    return {
        "rows": len(y_true),
        "malicious_rows": total_mal,
        "benign_rows": total_ben,
        "low_malicious_count": low_mal,
        "mal_leak_rate": low_mal / total_mal if total_mal else 0.0,
        "high_benign_count": high_ben,
        "high_benign_rate": high_ben / total_ben if total_ben else 0.0,
        "uncertain_count": int(np.sum(uncertain_mask)),
        "uncertain_rate": float(np.mean(uncertain_mask)),
        "auto_count": int(np.sum(auto_mask)),
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
        load_dedup_file(path, f"WEEK{i + 1}")
        for i, path in enumerate(files[:5])
    ]

    results = []

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

        if Xtr.shape[1] != 528 or Xev.shape[1] != 528:
            raise RuntimeError(
                f"Feature dimension mismatch: "
                f"train={Xtr.shape}, eval={Xev.shape}"
            )

        model: LGBMClassifier = build_model()
        model.fit(Xtr, ytr)

        # LightGBM의 predict_proba 반환 타입을 Pylance가
        # sparse matrix까지 포함한 union 타입으로 추론하는 문제를 피한다.
        raw_proba: Any = model.predict_proba(Xev)
        proba = cast(
            np.ndarray,
            np.asarray(raw_proba, dtype=np.float64),
        )

        if proba.ndim != 2 or proba.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected predict_proba shape: {proba.shape}"
            )

        scores = np.asarray(proba[:, 1], dtype=np.float64)

        m = routing_metrics(yev, scores)

        results.append(
            {
                "train_weeks": f"1-{eval_idx}",
                "eval_week": eval_idx + 1,
                "eval_file": files[eval_idx].name,
                "train_rows": len(ytr),
                **m,
            }
        )

    print("=" * 130)
    print("LIGHTGBM THRESHOLD STABILITY - EXPANDING WINDOW")
    print("=" * 130)
    print(
        f"LOW={LOW_THRESHOLD:.8f}, "
        f"HIGH={HIGH_THRESHOLD:.8f}"
    )
    print("Test data: NOT USED")
    print()

    print(
        f"{'Eval':<6} | "
        f"{'Train':>8} | "
        f"{'Rows':>7} | "
        f"{'Low→Mal':>7} | "
        f"{'MalLeak%':>9} | "
        f"{'High→Ben':>8} | "
        f"{'HighBen%':>9} | "
        f"{'Uncertain%':>10} | "
        f"{'Auto%':>7} | "
        f"{'AutoAcc%':>9}"
    )
    print("-" * 130)

    for r in results:
        print(
            f"W{r['eval_week']:<5} | "
            f"{r['train_rows']:>8,} | "
            f"{r['rows']:>7,} | "
            f"{r['low_malicious_count']:>7,} | "
            f"{r['mal_leak_rate'] * 100:>8.4f}% | "
            f"{r['high_benign_count']:>8,} | "
            f"{r['high_benign_rate'] * 100:>8.4f}% | "
            f"{r['uncertain_rate'] * 100:>9.2f}% | "
            f"{r['auto_rate'] * 100:>6.2f}% | "
            f"{r['auto_accuracy'] * 100:>8.3f}%"
        )

    leak_rates = np.asarray(
        [r["mal_leak_rate"] for r in results],
        dtype=np.float64,
    )
    high_ben_rates = np.asarray(
        [r["high_benign_rate"] for r in results],
        dtype=np.float64,
    )
    uncertain_rates = np.asarray(
        [r["uncertain_rate"] for r in results],
        dtype=np.float64,
    )
    auto_accs = np.asarray(
        [r["auto_accuracy"] for r in results],
        dtype=np.float64,
    )

    print("\n" + "=" * 130)
    print("STABILITY SUMMARY")
    print("=" * 130)

    print(
        f"MalLeak range    : "
        f"{leak_rates.min() * 100:.4f}% ~ "
        f"{leak_rates.max() * 100:.4f}%"
    )
    print(
        f"HighBen range    : "
        f"{high_ben_rates.min() * 100:.4f}% ~ "
        f"{high_ben_rates.max() * 100:.4f}%"
    )
    print(
        f"Uncertain range  : "
        f"{uncertain_rates.min() * 100:.2f}% ~ "
        f"{uncertain_rates.max() * 100:.2f}%"
    )
    print(
        f"Uncertain std    : "
        f"{uncertain_rates.std() * 100:.2f}%p"
    )
    print(
        f"AutoAcc range    : "
        f"{np.nanmin(auto_accs) * 100:.3f}% ~ "
        f"{np.nanmax(auto_accs) * 100:.3f}%"
    )

    max_low_mal = max(
        r["low_malicious_count"]
        for r in results
    )
    max_high_ben = max(
        r["high_benign_count"]
        for r in results
    )
    uncertain_span = (
        uncertain_rates.max()
        - uncertain_rates.min()
    )

    if max_low_mal > 1 or max_high_ben > 49:
        verdict = "THRESHOLDS_NOT_STABLE"
    elif uncertain_span > 0.10:
        verdict = "SAFETY_OK_BUT_ROUTING_VOLATILE"
    else:
        verdict = "THRESHOLDS_REASONABLY_STABLE"

    print(f"FINAL VERDICT: {verdict}")

    out = (
        REPORT_DIR
        / "lightgbm_threshold_stability_expanding_window.csv"
    )

    with out.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:")
    print(out)


if __name__ == "__main__":
    main()
