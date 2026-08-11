import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for Winner Bias analysis...")
df = pd.read_csv(os.path.join(script_dir, '..', 'eurovision_1957-2021.csv'))
df.columns = df.columns.str.strip()

# Clean and map points type
df['Points type'] = df['Points type'].str.strip().str.lower()
def map_point_type(pt):
    if 'jury' in pt:
        return 'Jury'
    elif 'televote' in pt:
        return 'Public'
    return 'Other'

df['Type'] = df['Points type'].apply(map_point_type)
df['To'] = df['To'].str.strip().str.title()

# We only want 2016 onwards where the 50/50 split voting exists explicitly in the data
df = df[(df['Year'] >= 2016) & (df['Type'] != 'Other')]

# Aggregate points received by each country per year and type
agg = df.groupby(['Year', 'To', 'Type'])['Points'].sum().reset_index()

# Pivot to get Jury and Public points as separate columns
pivot = agg.pivot_table(index=['Year', 'To'], columns='Type', values='Points', fill_value=0).reset_index()

# Calculate Total Points
pivot['Total'] = pivot['Jury'] + pivot['Public']

# Calculate Ranks (1 is highest points)
pivot['Rank_Total'] = pivot.groupby('Year')['Total'].rank(ascending=False, method='min')

# Get the winners (Rank_Total == 1)
winners = pivot[pivot['Rank_Total'] == 1].copy()
winners = winners.sort_values(by='Year', ascending=True)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#FAFAFA')
ax.set_facecolor('#FAFAFA')

years = winners['Year'].astype(str).tolist()
countries = winners['To'].tolist()
labels = [f"{y}\n{c}" for y, c in zip(years, countries)]

jury_pts = winners['Jury'].values
public_pts = winners['Public'].values
totals = winners['Total'].values

x = np.arange(len(labels))
width = 0.6

# Aesthetic colors
JURY_COLOR = '#2C3E50' # Dark Slate
PUBLIC_COLOR = '#E67E22' # Orange/Carrot

# Stacked Bar Chart
bar1 = ax.bar(x, jury_pts, width, label='Jury Points', color=JURY_COLOR, edgecolor='white', linewidth=1.5)
bar2 = ax.bar(x, public_pts, width, bottom=jury_pts, label='Public Points', color=PUBLIC_COLOR, edgecolor='white', linewidth=1.5)

# Add data labels
for i in range(len(x)):
    jp = jury_pts[i]
    pp = public_pts[i]
    tot = totals[i]
    
    j_pct = (jp / tot) * 100
    p_pct = (pp / tot) * 100
    
    # Jury text inside bar
    ax.text(x[i], jp / 2, f"{int(jp)}\n({j_pct:.1f}%)", ha='center', va='center', 
            color='white', fontweight='bold', fontsize=11)
            
    # Public text inside bar
    ax.text(x[i], jp + (pp / 2), f"{int(pp)}\n({p_pct:.1f}%)", ha='center', va='center', 
            color='white', fontweight='bold', fontsize=11)
            
    # Total text on top of bar
    ax.text(x[i], tot + 15, f"{int(tot)} pts", ha='center', va='bottom',
            color='#333333', fontweight='bold', fontsize=12)

# Styling
ax.set_title('Winning Bias: Jury vs. Public Points (2016-2021)', fontsize=22, fontweight='bold', color='#1A237E', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='#333333')
ax.set_ylabel('Total Points Awarded', fontsize=14, fontweight='bold', color='#333333')
ax.tick_params(axis='y', labelsize=11, colors='#333333')

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDC3C7')
ax.spines['bottom'].set_color('#BDC3C7')

ax.yaxis.grid(True, linestyle='--', color='#BDC3C7', alpha=0.7)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=12, framealpha=0.9, edgecolor='#BDC3C7')

plt.tight_layout()
out_path = os.path.join(output_dir, "winner_bias_timeline.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {out_path}")
