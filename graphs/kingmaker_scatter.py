import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for Kingmaker analysis...")
df = pd.read_csv(os.path.join(script_dir, '..', 'eurovision_1957-2021.csv'))
df.columns = df.columns.str.strip()

df['Points type'] = df['Points type'].str.strip().str.lower()
def map_point_type(pt):
    if 'jury' in pt: return 'Jury'
    elif 'televote' in pt: return 'Public'
    return 'Other'

df['Type'] = df['Points type'].apply(map_point_type)
df['To'] = df['To'].str.strip().str.title()
df = df[(df['Year'] >= 2016) & (df['Type'] != 'Other')]

agg = df.groupby(['Year', 'To', 'Type'])['Points'].sum().reset_index()
pivot = agg.pivot_table(index=['Year', 'To'], columns='Type', values='Points', fill_value=0).reset_index()
pivot['Total'] = pivot['Jury'] + pivot['Public']

pivot['Rank_Total'] = pivot.groupby('Year')['Total'].rank(ascending=False, method='min')
pivot['Rank_Jury'] = pivot.groupby('Year')['Jury'].rank(ascending=False, method='min')
pivot['Rank_Public'] = pivot.groupby('Year')['Public'].rank(ascending=False, method='min')

top10 = pivot[pivot['Rank_Total'] <= 10].copy()

# ── Colour palette ──────────────────────────────────────────────────────────
BG        = '#FAFBFC'
GRID_COL  = '#E0E4EC'
AXIS_COL  = '#5A6070'
ZONE_A    = '#E8F5E9'   # crowd + jury agree (top-left quad) — green tint
ZONE_B    = '#FFF3E0'   # crowd loved, jury didn't — amber tint
ZONE_C    = '#EDE7F6'   # jury loved, crowd didn't — purple tint
ZONE_D    = '#F5F5F5'   # neither — grey
WINNER_C  = '#F9A825'
WINNER_E  = '#E65100'
LOSER_C   = '#5C6BC0'
LOSER_E   = '#FFFFFF'
ANNO_WIN  = '#BF360C'
ANNO_LOSE = '#1A237E'
DIAGONAL  = '#B0BEC5'

fig, ax = plt.subplots(figsize=(13, 13), facecolor=BG)
ax.set_facecolor(BG)

# ── Quadrant shading  (boundary at rank 3) ──────────────────────────────────
split = 3.5
lim_lo, lim_hi = 0.3, 10.7

ax.fill_between([lim_lo, split], [lim_lo, lim_lo], [split, split], color=ZONE_A, zorder=0)   # top-left
ax.fill_between([split, lim_hi], [lim_lo, lim_lo], [split, split], color=ZONE_C, zorder=0)   # top-right
ax.fill_between([lim_lo, split], [split, split], [lim_hi, lim_hi], color=ZONE_B, zorder=0)   # bottom-left
ax.fill_between([split, lim_hi], [split, split], [lim_hi, lim_hi], color=ZONE_D, zorder=0)   # bottom-right

# Quadrant labels
quad_style = dict(fontsize=11, fontstyle='italic', ha='center', va='center', alpha=0.65)
ax.text(2.0,  2.0, "Both loved it\n(Likely winner)",      color='#2E7D32', **quad_style)
ax.text(7.0,  2.0, "Jury's pick\n(Crowd indifferent)",     color='#4527A0', **quad_style)
ax.text(2.0,  7.5, "Crowd's pick\n(Jury indifferent)",     color='#E65100', **quad_style)
ax.text(7.0,  7.5, "Neither loved it",                      color='#546E7A', **quad_style)

# ── Diagonal: perfect agreement line ───────────────────────────────────────
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=DIAGONAL, linestyle='--',
        linewidth=1.4, alpha=0.7, zorder=1, label='_nolegend_')
ax.text(9.0, 8.5, "Perfect\nagreement", fontsize=9, color=DIAGONAL,
        ha='center', rotation=45, style='italic')

# ── Scatter — non-winners ───────────────────────────────────────────────────
losers = top10[top10['Rank_Total'] > 1]
ax.scatter(losers['Rank_Public'], losers['Rank_Jury'],
           s=220, color=LOSER_C, edgecolors=LOSER_E,
           linewidths=1.5, alpha=0.75, zorder=2)

# ── Scatter — winners (star) ────────────────────────────────────────────────
winners = top10[top10['Rank_Total'] == 1]
ax.scatter(winners['Rank_Public'], winners['Rank_Jury'],
           s=800, marker='*', color=WINNER_C,
           edgecolors=WINNER_E, linewidths=1.5, zorder=3)

# ── Annotations ─────────────────────────────────────────────────────────────
used_positions = []

def get_offset(x, y, used):
    candidates = [
        (0.45, -0.25), (-0.15, -0.45), (0.45, 0.25),
        (-0.65, 0.0),  (0.0,   0.5),   (-0.15, 0.5),
    ]
    for dx, dy in candidates:
        nx, ny = x + dx, y + dy
        if not any(abs(nx - ux) < 1.1 and abs(ny - uy) < 0.5 for ux, uy in used):
            return dx, dy
    return candidates[0]

for _, row in winners.iterrows():
    label = f"{row['To']} '{str(int(row['Year']))[-2:]}"
    dx, dy = get_offset(row['Rank_Public'], row['Rank_Jury'], used_positions)
    used_positions.append((row['Rank_Public'] + dx, row['Rank_Jury'] + dy))
    ax.annotate(label,
                xy=(row['Rank_Public'], row['Rank_Jury']),
                xytext=(row['Rank_Public'] + dx, row['Rank_Jury'] + dy),
                fontsize=10.5, fontweight='bold', color=ANNO_WIN,
                arrowprops=dict(arrowstyle='->', color=ANNO_WIN, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=ANNO_WIN, alpha=0.85, lw=0.8))

for _, row in losers.iterrows():
    if row['Rank_Jury'] <= 2 or row['Rank_Public'] <= 2:
        label = f"{row['To']} '{str(int(row['Year']))[-2:]}"
        dx, dy = get_offset(row['Rank_Public'], row['Rank_Jury'], used_positions)
        used_positions.append((row['Rank_Public'] + dx, row['Rank_Jury'] + dy))
        ax.annotate(label,
                    xy=(row['Rank_Public'], row['Rank_Jury']),
                    xytext=(row['Rank_Public'] + dx, row['Rank_Jury'] + dy),
                    fontsize=9.5, fontweight='bold', color=ANNO_LOSE,
                    arrowprops=dict(arrowstyle='->', color=ANNO_LOSE, lw=0.9),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=ANNO_LOSE, alpha=0.8, lw=0.7))

# ── Axes ────────────────────────────────────────────────────────────────────
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_hi, lim_lo)   # inverted: rank 1 at top
ax.set_xticks(range(1, 11))
ax.set_yticks(range(1, 11))
ax.tick_params(colors=AXIS_COL, labelsize=12)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)

ax.set_xlabel('← Better Placement          Public Vote Rank (1 = Top of Public)          Worse Placement →',
              fontsize=12, fontweight='bold', color=AXIS_COL, labelpad=14)
ax.set_ylabel('← Worse Placement          Jury Vote Rank (1 = Top of Jury)          Better Placement →',
              fontsize=12, fontweight='bold', color=AXIS_COL, labelpad=14)

ax.xaxis.grid(True, linestyle='-', color=GRID_COL, linewidth=1.0, zorder=0)
ax.yaxis.grid(True, linestyle='-', color=GRID_COL, linewidth=1.0, zorder=0)

# ── Boundary lines between quadrants ────────────────────────────────────────
ax.axvline(split, color='#90A4AE', linewidth=1.2, linestyle=':', zorder=1)
ax.axhline(split, color='#90A4AE', linewidth=1.2, linestyle=':', zorder=1)

# ── Legend ──────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='*', color='w', label='Overall Winner',
           markerfacecolor=WINNER_C, markeredgecolor=WINNER_E, markersize=16),
    Line2D([0], [0], marker='o', color='w', label='Top 10 finisher',
           markerfacecolor=LOSER_C, markeredgecolor='white', markersize=10),
]
ax.legend(handles=legend_elements, loc='lower right',
          frameon=True, facecolor='white', edgecolor=GRID_COL,
          fontsize=12, labelcolor='#333333', framealpha=0.95)

# ── Title ────────────────────────────────────────────────────────────────────
ax.set_title('Who Decided? The Jury or the Public?\nContest Placement by Jury vs. Public Vote — Eurovision 2016–2021',
             fontsize=22, fontweight='bold', color='#1A237E', pad=22, linespacing=1.5)

# ── Insight banner ───────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Every winner (2016-2021) ranked in the Public's Top 2",
         ha='center', fontsize=12, color='#BF360C', fontweight='bold',
         bbox=dict(facecolor='#FFF8E1', edgecolor='#F9A825',
                   boxstyle='round,pad=0.6', alpha=0.95))

plt.tight_layout(rect=[0, 0.04, 1, 1])

filepath = os.path.join(output_dir, "kingmaker_scatter.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()

print(f"Chart saved successfully to: {filepath}")
