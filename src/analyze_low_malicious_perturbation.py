import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")

LOW_THRESHOLD = 0.03
CORE_DIM = 16
HIST_DIM = 256
BYTE_DIM = 256

CORE_NAMES = [
    "has_certificate",
    "has_pe_warning",
    "general_entropy",
    "import_api_count",
    "section_missing",
    "section_count",
    "mean_section_entropy",
    "max_section_entropy",
    "overlay_size",
    "overlay_size_ratio",
    "overlay_entropy",
    "file_size",
    "numstrings",
    "string_avlength",
    "string_entropy",
    "printables",
]


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


def renormalize_group(x):
    x = np.asarray(x, dtype=np.float32).copy()
    x[x < 0] = 0.0

    s = float(x.sum())

    if s <= 0:
        return np.zeros_like(x)

    return x / s


def predict_one(model, x):
    return float(model.predict_proba(x.reshape(1, -1))[0, 1])


def feature_name(global_index):
    if global_index < CORE_DIM:
        return CORE_NAMES[global_index], "CORE", global_index

    if global_index < CORE_DIM + HIST_DIM:
        local_index = global_index - CORE_DIM
        return f"histogram_{local_index:03d}", "HIST", local_index

    local_index = global_index - CORE_DIM - HIST_DIM
    return f"byteentropy_{local_index:03d}", "BYTE", local_index


def perturb_one_feature(base, global_index, reference_value):
    x = base.copy()

    if global_index < CORE_DIM:
        x[global_index] = reference_value
        return x

    if global_index < CORE_DIM + HIST_DIM:
        start = CORE_DIM
        end = CORE_DIM + HIST_DIM
        local_index = global_index - start

        group = x[start:end].copy()
        group[local_index] = reference_value
        x[start:end] = renormalize_group(group)

        return x

    start = CORE_DIM + HIST_DIM
    end = start + BYTE_DIM
    local_index = global_index - start

    group = x[start:end].copy()
    group[local_index] = reference_value
    x[start:end] = renormalize_group(group)

    return x


def perturb_group(base, group_name, reference):
    x = base.copy()

    if group_name == "CORE":
        x[:CORE_DIM] = reference[:CORE_DIM]

    elif group_name == "HIST":
        start = CORE_DIM
        end = CORE_DIM + HIST_DIM
        x[start:end] = renormalize_group(reference[start:end])

    elif group_name == "BYTE":
        start = CORE_DIM + HIST_DIM
        end = start + BYTE_DIM
        x[start:end] = renormalize_group(reference[start:end])

    else:
        raise ValueError(f"Unknown group: {group_name}")

    return x


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*_Win32_train.jsonl"))

    if len(files) < 5:
        print("[ERROR] JSONL 5개 이상 필요")
        return

    train = load_dedup(files[:4], "TRAIN")
    val = load_dedup([files[4]], "VAL")

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

    print(f"Train unique      : {len(ytr):,}")
    print(f"Validation unique : {len(yv):,}")

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

    target = np.where(
        (yv == 1) & (scores < LOW_THRESHOLD)
    )[0]

    low_ben = np.where(
        (yv == 0) & (scores < LOW_THRESHOLD)
    )[0]

    print(f"Low malicious (<{LOW_THRESHOLD:.2f}): {len(target):,}")
    print(f"Low benign    (<{LOW_THRESHOLD:.2f}): {len(low_ben):,}")

    if len(target) == 0:
        print("[INFO] 분석 대상 Low 악성이 없습니다.")
        return

    if len(low_ben) == 0:
        print("[ERROR] Low benign reference가 없습니다.")
        return

    # 정상 저위험군의 중앙값을 perturbation reference로 사용.
    # 평균보다 극단값의 영향을 덜 받고,
    # binary Core feature는 실제 0/1 값으로 유지된다.
    reference = np.median(Xv[low_ben], axis=0).astype(np.float32)

    # Histogram / ByteEntropy는 전체 합이 1이 되도록 정규화.
    hist_start = CORE_DIM
    hist_end = CORE_DIM + HIST_DIM
    byte_start = hist_end
    byte_end = byte_start + BYTE_DIM

    reference[hist_start:hist_end] = renormalize_group(
        reference[hist_start:hist_end]
    )
    reference[byte_start:byte_end] = renormalize_group(
        reference[byte_start:byte_end]
    )

    feature_rows = []
    group_rows = []

    print("=" * 120)
    print("LOW MALICIOUS FEATURE PERTURBATION")
    print("=" * 120)
    print(
        "delta_score = perturbed_score - original_score "
        "(양수면 해당 Feature를 정상 기준값으로 바꿨을 때 악성 점수가 상승)"
    )
    print("=" * 120)

    for idx in target:
        base = Xv[idx].copy()
        original_score = float(scores[idx])

        sample_rows = []

        for global_index in range(base.shape[0]):
            name, group, local_index = feature_name(global_index)

            perturbed = perturb_one_feature(
                base,
                global_index,
                reference[global_index],
            )

            new_score = predict_one(model, perturbed)
            delta = new_score - original_score

            row = {
                "validation_index": int(idx),
                "sha256": val[idx]["sha256"],
                "original_score": original_score,
                "feature_group": group,
                "feature_name": name,
                "feature_index": int(local_index),
                "original_value": float(base[global_index]),
                "reference_value": float(reference[global_index]),
                "perturbed_score": new_score,
                "delta_score": delta,
                "crossed_low_threshold": int(
                    original_score < LOW_THRESHOLD
                    and new_score >= LOW_THRESHOLD
                ),
            }

            feature_rows.append(row)
            sample_rows.append(row)

        group_results = []

        for group_name in ("CORE", "HIST", "BYTE"):
            perturbed = perturb_group(
                base,
                group_name,
                reference,
            )

            new_score = predict_one(model, perturbed)
            delta = new_score - original_score

            grow = {
                "validation_index": int(idx),
                "sha256": val[idx]["sha256"],
                "original_score": original_score,
                "feature_group": group_name,
                "perturbed_score": new_score,
                "delta_score": delta,
                "crossed_low_threshold": int(
                    original_score < LOW_THRESHOLD
                    and new_score >= LOW_THRESHOLD
                ),
            }

            group_rows.append(grow)
            group_results.append(grow)

        ranked = sorted(
            sample_rows,
            key=lambda r: r["delta_score"],
            reverse=True,
        )

        print(
            f"\nidx={idx:>6} | original={original_score:.6f} "
            f"| sha={val[idx]['sha256'][:12]}..."
        )

        print("  TOP +DELTA FEATURES")

        for rank, row in enumerate(ranked[:10], 1):
            print(
                f"  {rank:>2}. "
                f"{row['feature_name']:<24} "
                f"group={row['feature_group']:<4} "
                f"delta={row['delta_score']:+.6f} "
                f"score={row['perturbed_score']:.6f} "
                f"cross={row['crossed_low_threshold']}"
            )

        print("  GROUP PERTURBATION")

        for row in sorted(
            group_results,
            key=lambda r: r["delta_score"],
            reverse=True,
        ):
            print(
                f"      {row['feature_group']:<4} "
                f"delta={row['delta_score']:+.6f} "
                f"score={row['perturbed_score']:.6f} "
                f"cross={row['crossed_low_threshold']}"
            )

    feature_out = (
        REPORT_DIR
        / "dedup_low_malicious_feature_perturbation.csv"
    )

    group_out = (
        REPORT_DIR
        / "dedup_low_malicious_group_perturbation.csv"
    )

    save_csv(feature_out, feature_rows)
    save_csv(group_out, group_rows)

    # 전체 11개에서 어떤 Feature가 반복적으로 큰 영향을 보였는지 집계
    summary = {}

    for row in feature_rows:
        key = (
            row["feature_group"],
            row["feature_name"],
        )

        item = summary.setdefault(
            key,
            {
                "feature_group": row["feature_group"],
                "feature_name": row["feature_name"],
                "sample_count": 0,
                "positive_delta_count": 0,
                "threshold_cross_count": 0,
                "sum_delta": 0.0,
                "max_delta": float("-inf"),
            },
        )

        item["sample_count"] += 1

        if row["delta_score"] > 0:
            item["positive_delta_count"] += 1

        item["threshold_cross_count"] += int(
            row["crossed_low_threshold"]
        )

        item["sum_delta"] += float(row["delta_score"])
        item["max_delta"] = max(
            item["max_delta"],
            float(row["delta_score"]),
        )

    summary_rows = []

    for item in summary.values():
        n = item["sample_count"]

        summary_rows.append(
            {
                "feature_group": item["feature_group"],
                "feature_name": item["feature_name"],
                "sample_count": n,
                "positive_delta_count": item["positive_delta_count"],
                "threshold_cross_count": item["threshold_cross_count"],
                "mean_delta_score": item["sum_delta"] / n,
                "max_delta_score": item["max_delta"],
            }
        )

    summary_rows.sort(
        key=lambda r: (
            r["threshold_cross_count"],
            r["mean_delta_score"],
        ),
        reverse=True,
    )

    summary_out = (
        REPORT_DIR
        / "dedup_low_malicious_feature_perturbation_summary.csv"
    )

    save_csv(summary_out, summary_rows)

    print("\n" + "=" * 120)
    print("OVERALL TOP FEATURES")
    print("=" * 120)

    for rank, row in enumerate(summary_rows[:20], 1):
        print(
            f"{rank:>2}. "
            f"{row['feature_name']:<24} "
            f"group={row['feature_group']:<4} "
            f"mean_delta={row['mean_delta_score']:+.6f} "
            f"cross={row['threshold_cross_count']}/{row['sample_count']}"
        )

    print("\nSaved:")
    print(feature_out)
    print(group_out)
    print(summary_out)


if __name__ == "__main__":
    main()
