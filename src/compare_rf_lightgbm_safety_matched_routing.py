import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")

CORE_DIM = 16
HIST_DIM = 256

TARGET_MAX_LOW_MAL = 1
TARGET_MAX_HIGH_BEN = 49


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


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
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
        ),
    }


def find_low_threshold(y_true, scores, max_low_mal):
    mal_scores = np.sort(scores[y_true == 1])

    if max_low_mal <= 0:
        return float(np.nextafter(mal_scores[0], -np.inf))

    if len(mal_scores) <= max_low_mal:
        return 1.0

    # scores < threshold 가 low.
    # 정확히 max_low_mal 이하가 되도록 다음 악성 score 바로 아래를 사용.
    cutoff = mal_scores[max_low_mal]
    return float(np.nextafter(cutoff, -np.inf))


def find_high_threshold(y_true, scores, max_high_ben):
    ben_scores = np.sort(scores[y_true == 0])[::-1]

    if max_high_ben <= 0:
        return float(np.nextafter(ben_scores[0], np.inf))

    if len(ben_scores) <= max_high_ben:
        return 0.0

    # scores >= threshold 가 high.
    # 정확히 max_high_ben 이하가 되도록 (max_high_ben+1)번째 정상 score 바로 위 사용.
    cutoff = ben_scores[max_high_ben]
    return float(np.nextafter(cutoff, np.inf))


def routing_metrics(y_true, scores, low, high):
    low_mask = scores < low
    high_mask = scores >= high
    uncertain_mask = ~(low_mask | high_mask)
    auto_mask = low_mask | high_mask

    low_mal = int(np.sum((y_true == 1) & low_mask))
    high_ben = int(np.sum((y_true == 0) & high_mask))

    auto_true = y_true[auto_mask]
    auto_pred = np.where(scores[auto_mask] >= high, 1, 0)

    auto_acc = (
        float(accuracy_score(auto_true, auto_pred))
        if len(auto_true) > 0
        else float("nan")
    )

    total_mal = int(np.sum(y_true == 1))
    total_ben = int(np.sum(y_true == 0))

    return {
        "low_threshold": low,
        "high_threshold": high,
        "low_malicious_count": low_mal,
        "high_benign_count": high_ben,
        "mal_leak_rate": low_mal / total_mal if total_mal else 0.0,
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

    train = load_dedup(files[:4], "TRAIN")
    val = load_dedup([files[4]], "VAL")

    Xtr = np.asarray([vec(x["record"]) for x in train], dtype=np.float32)
    ytr = np.asarray([x["label"] for x in train], dtype=np.int8)

    Xv = np.asarray([vec(x["record"]) for x in val], dtype=np.float32)
    yv = np.asarray([x["label"] for x in val], dtype=np.int8)

    models = build_models()
    rows = []

    print("=" * 120)
    print("RF vs LightGBM - SAFETY MATCHED ROUTING COMPARISON")
    print("=" * 120)
    print(f"Target safety: Low→Mal <= {TARGET_MAX_LOW_MAL}, High→Ben <= {TARGET_MAX_HIGH_BEN}")
    print("Test data: NOT USED")
    print()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xv)[:, 1]

        low = find_low_threshold(
            y_true=yv,
            scores=scores,
            max_low_mal=TARGET_MAX_LOW_MAL,
        )
        high = find_high_threshold(
            y_true=yv,
            scores=scores,
            max_high_ben=TARGET_MAX_HIGH_BEN,
        )

        if low >= high:
            raise RuntimeError(
                f"{name}: low threshold >= high threshold "
                f"({low:.8f} >= {high:.8f})"
            )

        r = routing_metrics(
            y_true=yv,
            scores=scores,
            low=low,
            high=high,
        )

        row = {
            "model": name,
            **r,
        }
        rows.append(row)

        print(
            f"{name:<13} | "
            f"LOW={low:.8f} | "
            f"HIGH={high:.8f} | "
            f"Low→Mal={r['low_malicious_count']:>3} | "
            f"High→Ben={r['high_benign_count']:>3} | "
            f"Uncertain={r['uncertain_count']:>6,} "
            f"({r['uncertain_rate']*100:>6.2f}%) | "
            f"Auto={r['auto_count']:>6,} "
            f"({r['auto_rate']*100:>6.2f}%) | "
            f"AutoAcc={r['auto_accuracy']*100:>7.3f}%"
        )

    print("\n" + "=" * 120)
    print("SAFETY-MATCHED RESULT")
    print("=" * 120)

    ranked = sorted(
        rows,
        key=lambda r: (
            r["uncertain_rate"],
            -r["auto_accuracy"],
        ),
    )

    for rank, row in enumerate(ranked, 1):
        print(
            f"{rank}. {row['model']}: "
            f"Uncertain={row['uncertain_rate']*100:.2f}% | "
            f"Auto={row['auto_rate']*100:.2f}% | "
            f"Low→Mal={row['low_malicious_count']} | "
            f"High→Ben={row['high_benign_count']} | "
            f"AutoAcc={row['auto_accuracy']*100:.3f}% | "
            f"LOW={row['low_threshold']:.8f} | "
            f"HIGH={row['high_threshold']:.8f}"
        )

    winner = ranked[0]

    print()
    print(f"Best routing efficiency under same safety limits: {winner['model']}")

    out = REPORT_DIR / "rf_vs_lightgbm_safety_matched_routing.csv"

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
