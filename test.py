import numpy as np
import pandas as pd

def generate_sm1_idle_run_segments(
    n_rows=8000,
    seed=42,
    idle_range=(14, 25),
    run_range=(1491, 1558),
    avg_segment_len=250,     # 구간 평균 길이 (크면 더 긴 덩어리)
    idle_ratio=0.15,         # 전체 중 비가동 비율(대략)
    out_csv="sm1_fake.csv"
):
    rng = np.random.default_rng(seed)

    idx = np.arange(1, n_rows + 1)
    sm1 = np.empty(n_rows, dtype=int)

    # 구간 길이: 평균 avg_segment_len 근처로 나오게(지수분포 기반)
    def sample_seg_len():
        # 너무 짧거나 너무 길지 않게 클램프
        L = int(rng.exponential(avg_segment_len)) + 20
        return int(np.clip(L, 30, 800))

    pos = 0
    while pos < n_rows:
        seg_len = sample_seg_len()
        seg_len = min(seg_len, n_rows - pos)

        # idle_ratio 확률로 비가동, 아니면 가동
        is_idle = (rng.random() < idle_ratio)

        if is_idle:
            sm1[pos:pos+seg_len] = rng.integers(idle_range[0], idle_range[1] + 1, size=seg_len)
        else:
            sm1[pos:pos+seg_len] = rng.integers(run_range[0], run_range[1] + 1, size=seg_len)

        pos += seg_len

    df = pd.DataFrame({"idx": idx, "SM1": sm1})
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv} (rows={len(df)})")
    return df

if __name__ == "__main__":
    generate_sm1_idle_run_segments(
        n_rows=8000,
        seed=42,
        avg_segment_len=250,   # 구간 평균 길이 조절 가능
        idle_ratio=0.25,       # 비가동 비율 조절 가능
        out_csv="sm1_fake.csv"
    )
