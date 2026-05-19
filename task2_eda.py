import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

def get_ts(df, area):
    mask = df['square_id'] == area
    ts = (df[mask]
          .groupby('datetime')['internet']
          .sum()
          .asfreq('10min')
          .fillna(0))
    return ts
# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT   = r'C:\Users\HP\Downloads\milan_traffic\traffic_optimized.parquet'
PLOTS   = r'C:\Users\HP\Downloads\milan_traffic\plots'
import os; os.makedirs(PLOTS, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(INPUT)
print(f"Loaded: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 1: PDF of total traffic per area
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] PDF of total traffic per area...")
total_per_area = df.groupby('square_id')['internet'].sum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(total_per_area.values, bins=100, density=True,
        color='steelblue', edgecolor='white', alpha=0.85)
ax.set_xlabel('Total 2-Month Internet Traffic', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('PDF of Total Traffic per Geographical Area', fontsize=14)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{PLOTS}/1_pdf_traffic.png', dpi=150)
plt.close()
print("  Saved: 1_pdf_traffic.png")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 2: Identify the 3 target areas
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Identifying target areas...")
highest_area = total_per_area.idxmax()
print(f"  Highest traffic area: Square ID {highest_area} "
      f"(total={total_per_area[highest_area]:.1f})")
print(f"  Fixed areas: 4159, 4556")

TARGET_AREAS = [highest_area, 4159, 4556]
AREA_LABELS  = [f'Area {highest_area} (Highest)', 'Area 4159', 'Area 4556']

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 3: Time series — first two weeks, 3 areas
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Time series plots (first 2 weeks)...")
TWO_WEEKS_END = df['datetime'].min() + pd.Timedelta(weeks=2)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
colors = ['steelblue', 'darkorange', 'seagreen']

for ax, area, label, color in zip(axes, TARGET_AREAS, AREA_LABELS, colors):
    ts = get_ts(df, area)
    ax.plot(ts.index, ts.values, color=color, linewidth=0.8)
    ax.set_ylabel('Traffic', fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Date', fontsize=11)
fig.suptitle('Network Traffic — First Two Weeks (3 Areas)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOTS}/2_timeseries_2weeks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 2_timeseries_2weeks.png")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 4: Stationarity — rolling stats + ADF test
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Stationarity analysis...")

def adf_test(series, label):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n  ADF Test — {label}")
    print(f"    ADF Statistic : {result[0]:.4f}")
    print(f"    p-value       : {result[1]:.4f}")
    print(f"    Stationary    : {'YES ✓' if result[1] < 0.05 else 'NO ✗'}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
WINDOW = 144  # 1 day (144 × 10min = 1440 min)

for ax, area, label, color in zip(axes, TARGET_AREAS, AREA_LABELS, colors):
    ts = get_ts(df, area)
    
    roll_mean = ts.rolling(WINDOW).mean()
    roll_std  = ts.rolling(WINDOW).std()
    
    ax.plot(ts.index, ts.values,       color=color,  alpha=0.4, linewidth=0.6, label='Traffic')
    ax.plot(roll_mean.index, roll_mean, color='black', linewidth=1.2, label='Rolling Mean (1d)')
    ax.plot(roll_std.index,  roll_std,  color='red',   linewidth=1.0, linestyle='--', label='Rolling Std (1d)')
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    adf_test(ts, label)

fig.suptitle('Rolling Statistics & Stationarity', fontsize=14)
plt.tight_layout()
plt.savefig(f'{PLOTS}/3_stationarity.png', dpi=150)
plt.close()
print("\n  Saved: 3_stationarity.png")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 5: Seasonal decomposition
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Seasonal decomposition...")

# Use highest traffic area, one month of data
area = TARGET_AREAS[0]
ts = get_ts(df, area)
ts_month = ts.iloc[:4032]  # ~4 weeks (4032 × 10min)

# Period = 1 day = 144 intervals
decomp = seasonal_decompose(ts_month, model='additive', period=144)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], color='steelblue');  axes[0].set_ylabel('Observed')
decomp.trend.plot(ax=axes[1],    color='darkorange'); axes[1].set_ylabel('Trend')
decomp.seasonal.plot(ax=axes[2], color='seagreen');   axes[2].set_ylabel('Seasonal')
decomp.resid.plot(ax=axes[3],    color='gray');       axes[3].set_ylabel('Residual')
for ax in axes: ax.grid(True, alpha=0.3)
fig.suptitle(f'Seasonal Decomposition — Area {area}', fontsize=14)
plt.tight_layout()
plt.savefig(f'{PLOTS}/4_decomposition.png', dpi=150)
plt.close()
print("  Saved: 4_decomposition.png")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 6: ACF & PACF
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] ACF & PACF plots...")
ts_acf = ts.iloc[:2016]  # 2 weeks for speed

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
plot_acf(ts_acf,  ax=axes[0], lags=288, alpha=0.05)  # 288 = 2 days
plot_pacf(ts_acf, ax=axes[1], lags=100, alpha=0.05, method='ywm')
axes[0].set_title(f'ACF — Area {area}')
axes[1].set_title(f'PACF — Area {area}')
for ax in axes: ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/5_acf_pacf.png', dpi=150)
plt.close()
print("  Saved: 5_acf_pacf.png")

# ═══════════════════════════════════════════════════════════════════════════════
# EDA 7: Spatial heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Spatial heatmap...")
grid = np.zeros((100, 100))

for sid, val in total_per_area.items():
    sid = int(sid)
    row = (sid - 1) // 100   # 0-indexed row
    col = (sid - 1) % 100    # 0-indexed col
    grid[row, col] = val

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.log1p(grid), cmap='hot', origin='lower', aspect='equal')
plt.colorbar(im, ax=ax, label='log(1 + Total Traffic)')
ax.set_title('Spatial Heatmap of Total Traffic (log scale)', fontsize=14)
ax.set_xlabel('Grid Column')
ax.set_ylabel('Grid Row')

# Mark the 3 target areas
for area, label in zip(TARGET_AREAS, AREA_LABELS):
    area = int(area)
    row = (area - 1) // 100
    col = (area - 1) % 100
    ax.plot(col, row, 'c*', markersize=12)
    ax.annotate(label, (col, row), textcoords='offset points',
                xytext=(5, 5), color='cyan', fontsize=8)

plt.tight_layout()
plt.savefig(f'{PLOTS}/6_heatmap.png', dpi=150)
plt.close()
print("  Saved: 6_heatmap.png")

print("\n✅ All EDA plots saved to:", PLOTS)