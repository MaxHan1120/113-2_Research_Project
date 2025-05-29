import argparse, numpy as np, pandas as pd
from pathlib import Path

# -------- CLI -------- #
ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="CSV 檔 (time,value)")
ap.add_argument("--f0", type=float, default=18370, help="名目頻率 Hz")
ap.add_argument("--th", type=float, default=0.5, help="門檻電壓 (交點)")
ap.add_argument("--gate_guess", type=float, default=None,
                help="單週期粗估 (s)，若留空用 1/f0")
ap.add_argument("--tcol", type=int, default=0)
ap.add_argument("--vcol", type=int, default=1)
args = ap.parse_args()

# -------- 讀入資料 -------- #
csv = Path(args.file)
df = pd.read_csv(csv)
t = df[df.columns[0]].astype(float).values
v = df[df.columns[1]].astype(float).values

del df

# -------- 找交點 (v<thr -> v>=thr) -------- #
thr = args.th
cross_idx = np.where((v[:-1] < thr) & (v[1:] >= thr))[0] + 1
t_cross = t[cross_idx]

if len(t_cross) < 3:
    raise RuntimeError("交點過少，請檢查門檻或波形")

# -------- 計算週期 -------- #
T = np.diff(t_cross)
T0 = 1.0 / args.f0
est_T = args.gate_guess or T0
mask = np.abs(T - est_T) < est_T * 0.2
T, t_cross = T[mask], t_cross[:-1][mask]

# 儲存週期與交點
np.savez_compressed(csv.with_suffix(".periods.npz"), T=T, t_cross=t_cross, T0=T0)
print(f"[✓] 週期序列已儲存至 {csv.with_suffix('.periods.npz')}")
