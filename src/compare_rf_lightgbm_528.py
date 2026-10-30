import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")

LOW_THRESHOLD = 0.01
HIGH_THRESHOLD = 0.85

CORE_DIM = 16
HIST_DIM = 256
BYTE_DIM = 256


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


def binary_metrics(y_true, scores):
    pred = (scores >= 0.5).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(
            average_precision_score(y_true, scores)
        ),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def routing_metrics(y_true, scores, low, high):
    low_mask = scores < low
    high_mask = scores >= high
    uncertain_mask = ~(low_mask | high_mask)
    auto_mask = low_mask | high_mask

    low_mal = int(
        np.sum((y_true == 1) & low_mask)
    )
    low_ben = int(
        np.sum((y_true == 0) & low_mask)
    )
    high_mal = int(
        np.sum((y_true == 1) & high_mask)
    )
    high_ben = int(
        np.sum((y_true == 0) & high_mask)
    )

    total_mal = int(np.sum(y_true == 1))
    total_ben = int(np.sum(y_true == 0))

    auto_true = y_true[auto_mask]
    auto_pred = np.where(
        scores[auto_mask] >= high,
        1,
        0,
    )

    auto_acc = (
        float(accuracy_score(auto_true, auto_pred))
        if len(auto_true)
        else float("nan")
    )

    return {
        "low_threshold": low,
        "high_threshold": high,
        "low_count": int(np.sum(low_mask)),
        "uncertain_count": int(np.sum(uncertain_mask)),
        "high_count": int(np.sum(high_mask)),
        "auto_count": int(np.sum(auto_mask)),
        "auto_rate": float(np.mean(auto_mask)),
        "uncertain_rate": float(np.mean(uncertain_mask)),
        "low_malicious_count": low_mal,
        "low_benign_count": low_ben,
        "mal_leak_rate": (
            low_mal / total_mal
            if total_mal
            else 0.0
        ),
        "high_malicious_count": high_mal,
        "high_benign_count": high_ben,
        "high_benign_rate": (
            high_ben / total_ben
            if total_ben
            else 0.0
        ),
        "auto_accuracy": auto_acc,
    }


def build_models():
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    # Baseline LightGBM:
    # 과도한 튜닝 없이 RF와 동일 528 Feature에서
    # 기본적인 gradient boosting 성능을 공정 비교한다.
    lgbm = LGBMClassifier(
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

    return {
        "RandomForest": rf,
        "LightGBM": lgbm,
    }


def print_binary(name, m):
    print(f"\n[{name}]")
    print("-" * 80)
    print(f"Accuracy  : {m['accuracy']:.6f}")
    print(f"Precision : {m['precision']:.6f}")
    print(f"Recall    : {m['recall']:.6f}")
    print(f"F1        : {m['f1']:.6f}")
    print(f"ROC-AUC   : {m['roc_auc']:.6f}")
    print(f"PR-AUC    : {m['pr_auc']:.6f}")
    print(f"FPR       : {m['fpr']:.6f}")
    print(f"FNR       : {m['fnr']:.6f}")
    print(
        f"TN={m['tn']:,} "
        f"FP={m['fp']:,} "
        f"FN={m['fn']:,} "
        f"TP={m['tp']:,}"
    )


def print_routing(name, r):
    print(
        f"{name:<13} | "
        f"Uncertain={r['uncertain_count']:>6,} "
        f"({r['uncertain_rate']*100:>6.2f}%) | "
        f"Auto={r['auto_count']:>6,} "
        f"({r['auto_rate']*100:>6.2f}%) | "
        f"Low→Mal={r['low_malicious_count']:>4,} | "
        f"MalLeak={r['mal_leak_rate']*100:>7.4f}% | "
        f"High→Ben={r['high_benign_count']:>4,} | "
        f"AutoAcc={r['auto_accuracy']*100:>7.3f}%"
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        DATA_DIR.glob("*_Win32_train.jsonl")
    )

    if len(files) < 5:
        print("[ERROR] JSONL 5개 이상 필요")
        return

    train = load_dedup(
        files[:4],
        "TRAIN",
    )
    val = load_dedup(
        [files[4]],
        "VAL",
    )

    Xtr = np.asarray(
        [vec(x["record"]) for x in train],
        dtype=np.float32,
    )
    ytr = np.asarray(
        [x["label"] for x in train],
        dtype=np.int8,
    )

    Xv = np.asarray(
        [vec(x["record"]) for x in val],
        dtype=np.float32,
    )
    yv = np.asarray(
        [x["label"] for x in val],
        dtype=np.int8,
    )

    if Xtr.shape[1] != 528 or Xv.shape[1] != 528:
        raise RuntimeError(
            f"Feature dimension mismatch: "
            f"train={Xtr.shape}, val={Xv.shape}"
        )

    print()
    print("=" * 100)
    print("RF vs LightGBM 528 FAIR COMPARISON")
    print("=" * 100)
    print(f"Train unique      : {len(ytr):,}")
    print(f"Validation unique : {len(yv):,}")
    print(f"Features          : {Xtr.shape[1]}")
    print("Test data         : NOT USED")
    print(
        f"Routing reference : "
        f"LOW={LOW_THRESHOLD:.2f}, "
        f"HIGH={HIGH_THRESHOLD:.2f}"
    )

    models = build_models()

    rows = []
    score_rows = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(Xtr, ytr)

        scores = model.predict_proba(Xv)[:, 1]

        bm = binary_metrics(
            yv,
            scores,
        )
        rm = routing_metrics(
            yv,
            scores,
            LOW_THRESHOLD,
            HIGH_THRESHOLD,
        )

        print_binary(
            name,
            bm,
        )

        rows.append(
            {
                "model": name,
                **bm,
                **rm,
            }
        )

        for idx, score in enumerate(scores):
            score_rows.append(
                {
                    "model": name,
                    "validation_index": idx,
                    "sha256": val[idx]["sha256"],
                    "label": int(yv[idx]),
                    "score": float(score),
                }
            )

    print("\n" + "=" * 100)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 100)
    print(
        f"{'Model':<13} | "
        f"{'ROC-AUC':>8} | "
        f"{'PR-AUC':>8} | "
        f"{'Recall':>8} | "
        f"{'FPR':>8} | "
        f"{'FNR':>8} | "
        f"{'FP':>6} | "
        f"{'FN':>6}"
    )
    print("-" * 100)

    for row in rows:
        print(
            f"{row['model']:<13} | "
            f"{row['roc_auc']:>8.5f} | "
            f"{row['pr_auc']:>8.5f} | "
            f"{row['recall']:>8.5f} | "
            f"{row['fpr']:>8.5f} | "
            f"{row['fnr']:>8.5f} | "
            f"{row['fp']:>6,} | "
            f"{row['fn']:>6,}"
        )

    print("\n" + "=" * 120)
    print("ROUTING COMPARISON - SAME THRESHOLDS")
    print("=" * 120)
    for row in rows:
        print_routing(
            row["model"],
            row,
        )

    # 모델 선택은 단일 지표가 아니라 보안 성능 우선 순위로 정렬.
    # PR-AUC -> ROC-AUC -> FNR 낮음 -> FPR 낮음 순으로 참고.
    ranked = sorted(
        rows,
        key=lambda r: (
            r["pr_auc"],
            r["roc_auc"],
            -r["fnr"],
            -r["fpr"],
        ),
        reverse=True,
    )

    winner = ranked[0]

    print("\n" + "=" * 100)
    print("BASELINE COMPARISON RESULT")
    print("=" * 100)
    print(
        f"Current best baseline: {winner['model']}"
    )
    print(
        "주의: 이 결과만으로 최종 모델 확정하지 않음. "
        "LightGBM이 우세하면 다음 단계에서 제한된 튜닝과 "
        "threshold 재검증을 진행."
    )

    metrics_out = (
        REPORT_DIR
        / "rf_vs_lightgbm_528_comparison.csv"
    )
    scores_out = (
        REPORT_DIR
        / "rf_vs_lightgbm_528_validation_scores.csv"
    )

    with metrics_out.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    with scores_out.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(score_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(score_rows)

    print("\nSaved:")
    print(metrics_out)
    print(scores_out)


if __name__ == "__main__":
    main()
