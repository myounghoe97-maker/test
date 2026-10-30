import csv, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\데이터셋\train_sample")
REPORT_DIR = Path(r"C:\Users\user\Desktop\최종프로젝트\malware_detection\reports")
LOW_THRESHOLD = 0.03
CORE_DIM, HIST_DIM = 16, 256

CORE_NAMES = [
    "has_certificate","has_pe_warning","general_entropy","import_api_count",
    "section_missing","section_count","mean_section_entropy","max_section_entropy",
    "overlay_size","overlay_size_ratio","overlay_entropy","file_size",
    "numstrings","string_avlength","string_entropy","printables",
]

def number(v, d=0.0):
    return float(v) if isinstance(v, (int,float)) else d

def get_sha256(r):
    for k in ("sha256","sha_256","hash"):
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
    import_count = sum(len(v) for v in imports.values() if isinstance(v, list))
    section = r.get("section")
    section_missing = 1.0 if not section else 0.0
    section = section or {}
    sections = section.get("sections") or []
    ents = [float(x["entropy"]) for x in sections if isinstance(x.get("entropy"), (int,float))]
    mean_ent = sum(ents)/len(ents) if ents else 0.0
    max_ent = max(ents) if ents else 0.0
    overlay = section.get("overlay") or {}
    return np.asarray([
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
    ], dtype=np.float32)

def norm256(r, key):
    vals = r.get(key)
    if not isinstance(vals, list) or len(vals) != 256:
        return np.zeros(256, dtype=np.float32)
    a = np.asarray(vals, dtype=np.float32)
    s = float(a.sum())
    return a/s if s > 0 else np.zeros(256, dtype=np.float32)

def vec(r):
    return np.concatenate([extract_core(r), norm256(r,"histogram"), norm256(r,"byteentropy")]).astype(np.float32)

def load_dedup(files, tag):
    out, seen = [], set()
    for p in files:
        print(f"[{tag}] {p.name}")
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("label") not in (0,1):
                    continue
                sha = get_sha256(r)
                if not sha or sha in seen:
                    continue
                seen.add(sha)
                out.append({"record":r,"sha256":sha,"label":r["label"],"line_number":line_no})
    return out

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

    print(f"Train unique      : {len(ytr):,}")
    print(f"Validation unique : {len(yv):,}")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=18, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, random_state=42
    )
    model.fit(Xtr, ytr)
    scores = model.predict_proba(Xv)[:,1]

    target = np.where((yv==1) & (scores < LOW_THRESHOLD))[0]
    low_ben = np.where((yv==0) & (scores < LOW_THRESHOLD))[0]
    all_mal = np.where(yv==1)[0]

    print(f"Low malicious (<0.03): {len(target):,}")
    if len(target) == 0:
        return

    core = Xv[:,:CORE_DIM]
    hist = Xv[:,CORE_DIM:CORE_DIM+HIST_DIM]
    byte = Xv[:,CORE_DIM+HIST_DIM:]

    lb_core_mean, lb_core_std = core[low_ben].mean(0), core[low_ben].std(0)
    am_core_mean, am_core_std = core[all_mal].mean(0), core[all_mal].std(0)
    lb_hist_mean, am_hist_mean = hist[low_ben].mean(0), hist[all_mal].mean(0)
    lb_byte_mean, am_byte_mean = byte[low_ben].mean(0), byte[all_mal].mean(0)

    rows = []
    print("="*120)
    print("DEDUP RF - LOW MALICIOUS ANALYSIS")
    print("="*120)
    for idx in target:
        c,h,b = core[idx], hist[idx], byte[idx]
        cdb = float(np.linalg.norm((c-lb_core_mean)/np.where(lb_core_std>1e-12, lb_core_std, 1.0)))
        cdm = float(np.linalg.norm((c-am_core_mean)/np.where(am_core_std>1e-12, am_core_std, 1.0)))
        hdb, hdm = float(np.linalg.norm(h-lb_hist_mean)), float(np.linalg.norm(h-am_hist_mean))
        bdb, bdm = float(np.linalg.norm(b-lb_byte_mean)), float(np.linalg.norm(b-am_byte_mean))
        cc = "BENIGN" if cdb < cdm else "MALICIOUS"
        hc = "BENIGN" if hdb < hdm else "MALICIOUS"
        bc = "BENIGN" if bdb < bdm else "MALICIOUS"
        print(f"idx={idx:>6} | score={scores[idx]:.6f} | CORE={cc:<9} | HIST={hc:<9} | BYTE={bc:<9}")
        rows.append({
            "validation_index":int(idx),"sha256":val[idx]["sha256"],"score":float(scores[idx]),
            "core_closer":cc,"hist_closer":hc,"byte_closer":bc,
            "core_dist_benign":cdb,"core_dist_malicious":cdm,
            "hist_dist_benign":hdb,"hist_dist_malicious":hdm,
            "byte_dist_benign":bdb,"byte_dist_malicious":bdm
        })

    print("="*120)
    print("CORE FEATURE DIRECTION")
    print("="*120)
    for j,name in enumerate(CORE_NAMES):
        lm = float(core[target,j].mean())
        lb = float(lb_core_mean[j])
        am = float(am_core_mean[j])
        closer = "BENIGN" if abs(lm-lb) < abs(lm-am) else "MALICIOUS"
        print(f"{name:<24} lowMal={lm:>10.4f} | lowBen={lb:>10.4f} | allMal={am:>10.4f} | closer={closer}")

    out = REPORT_DIR / "dedup_low_malicious_003_analysis.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out)

if __name__ == "__main__":
    main()
