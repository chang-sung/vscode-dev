import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
input_path = r"C:\path\to\input.csv"

EQ_COL   = "EQ_Name"
TIME_COL = "WriteTime"
target_eq = None
target_sm = "SM8"

row_start = 0
row_end   = 400

IDLE_TH = 500
GAP_SEC = 7200    # 1시간 주기 데이터 기준 추천

EPS = 1e-6
# =========================

def robust_norm_series(x, eps=1e-6):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + eps)

def split_runs_by_gap(times, max_gap_sec=None):
    n = len(times)
    run_id = np.zeros(n, dtype=np.int32)
    if n == 0 or max_gap_sec is None:
        return run_id

    t = pd.to_datetime(times)
    dt = (t[1:].values - t[:-1].values) / np.timedelta64(1, "s")

    rid = 0
    for i in range(1, n):
        if dt[i-1] > max_gap_sec:
            rid += 1
        run_id[i] = rid
    return run_id

# -------------------------
# 1) 로드 / 정렬
# -------------------------
df = pd.read_csv(input_path, encoding="utf-8-sig")
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()

if target_eq is None:
    target_eq = df[EQ_COL].dropna().iloc[0]

dfe = (
    df[df[EQ_COL] == target_eq]
    .sort_values(TIME_COL)
    .reset_index(drop=True)
)

# -------------------------
# 2) 구간 선택
# -------------------------
seg = dfe.iloc[row_start:row_end].copy()
seg[target_sm] = pd.to_numeric(seg[target_sm], errors="coerce").fillna(0)

# -------------------------
# 3) 정지 데이터 제거
# -------------------------
seg = seg[seg[target_sm] > IDLE_TH].copy()
seg = seg.sort_values(TIME_COL).reset_index(drop=True)

if len(seg) < 5:
    raise ValueError("정지 제거 후 데이터가 너무 적습니다.")

t = seg[TIME_COL].values
raw = seg[target_sm].astype(np.float32).values

# -------------------------
# 4) run 분리
# -------------------------
run_id = split_runs_by_gap(t, GAP_SEC)

norm = np.zeros_like(raw)
dx   = np.zeros_like(raw)
d2x  = np.zeros_like(raw)

for rid in np.unique(run_id):
    idx = np.where(run_id == rid)[0]
    r_raw = raw[idx]

    r_norm = robust_norm_series(r_raw, EPS)
    r_dx   = np.diff(r_norm, prepend=r_norm[0])
    r_d2x  = np.diff(r_dx, prepend=r_dx[0])

    r_dx[0] = 0.0
    r_d2x[0] = 0.0

    norm[idx] = r_norm
    dx[idx]   = r_dx
    d2x[idx]  = r_d2x

# -------------------------
# 5) 시각화 (scatter + subplot)
# -------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].scatter(t, raw, s=10)
axes[0].set_title(f"{target_eq} | {target_sm} RAW (>500)")
axes[0].set_ylabel("raw")

axes[1].scatter(t, norm, s=10)
axes[1].set_title("robust normalized (pattern)")
axes[1].set_ylabel("norm")

axes[2].scatter(t, dx, s=10)
axes[2].set_title("1st derivative (dx)")
axes[2].set_ylabel("dx")

axes[3].scatter(t, d2x, s=10)
axes[3].set_title("2nd derivative (d2x)")
axes[3].set_ylabel("d2x")
axes[3].set_xlabel("WriteTime")

plt.suptitle(
    f"{target_eq} | rows {row_start}~{row_end-1} | {target_sm}\n"
    f"(idle<=500 removed, gap_sec={GAP_SEC})",
    y=0.98
)

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()