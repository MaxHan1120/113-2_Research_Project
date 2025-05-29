import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# -------- CLI -------- #
ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True, help="輸入 npz 檔（T, t_cross, T0）")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()

data = np.load(args.npz)
T, T0 = data["T"], data["T0"]
y = -(T - T0) / T0

# -------- Allan deviation -------- #
def allan(y, m):
    if m < 2 or len(y) < m * 2: return np.nan
    cs = np.cumsum(np.insert(y, 0, 0.0))
    y_bar = (cs[m:] - cs[:-m]) / m
    return np.sqrt(0.5 * np.mean((y_bar[m:] - y_bar[:-m])**2))

m_list = np.unique(np.round(np.logspace(
    np.log10(1),
    np.log10(len(y) // 4),
    num=100
)).astype(int))

taus = m_list * T0
sigma = np.array([allan(y, m) for m in m_list])

# -------- 儲存與畫圖 -------- #
out_csv = Path(args.npz).with_name("allan_result.csv")
pd.DataFrame({"tau_s": taus, "sigma_y": sigma}).to_csv(out_csv, index=False)
print(f"[✓] Allan deviation 結果已儲存到 {out_csv}")

plt.figure(); plt.loglog(taus, sigma, 'o-')
plt.xlabel(r"Average Time $\tau$ (s)")
plt.ylabel(r"Allan Deviation $\sigma_y$")
plt.grid(which='both')
plt.title(Path(args.npz).stem)
plt.tight_layout()
png = Path(args.npz).with_suffix(".allan.png")
plt.savefig(png, dpi=300)
print(f"[✓] Allan deviation 圖檔已寫入 {png}")
if args.show: plt.show()
