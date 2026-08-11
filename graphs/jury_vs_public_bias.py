import pandas as pd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading data for Jury vs Public bias...")
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
pivot['Rank_Jury'] = pivot.groupby('Year')['Jury'].rank(ascending=False, method='min')
pivot['Rank_Public'] = pivot.groupby('Year')['Public'].rank(ascending=False, method='min')

# Get Top 5 countries per year based on Total Rank
top5 = pivot[pivot['Rank_Total'] <= 5].copy()
# Sort so the highest rank (1) is at the top for each year group
top5 = top5.sort_values(by=['Year', 'Rank_Total'], ascending=[False, False])

# Plotting
fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
ax.set_facecolor('white')

years = top5['Year'].unique()
y_positions = []
y_labels = []

current_y = 0
for year in years:
    year_data = top5[top5['Year'] == year]
    
    # We want rank 1 at the top of its block, so we iterate in reverse order of Rank_Total
    # But we already sorted it ascending=[False, False], which means Rank 5 is first.
    # So we plot Rank 5, then Rank 4, ... moving UP the y-axis, which is correct.
    
    for _, row in year_data.iterrows():
        y_positions.append(current_y)
        y_labels.append(row['To'])
        
        r_tot = row['Rank_Total']
        r_jury = row['Rank_Jury']
        r_pub = row['Rank_Public']
        
        # Draw line connecting min and max rank
        min_r = min(r_tot, r_jury, r_pub)
        max_r = max(r_tot, r_jury, r_pub)
        ax.plot([min_r, max_r], [current_y, current_y], color='#d3d3d3', zorder=1, linewidth=4)
        
        # Draw points
        ax.scatter(r_pub, current_y, color='#00d2ff', s=200, zorder=3, alpha=0.9, label='Public Vote' if current_y == 0 else "")
        ax.scatter(r_jury, current_y, color='#ff007f', s=200, zorder=3, alpha=0.9, label='Jury Vote' if current_y == 0 else "")
        ax.scatter(r_tot, current_y, color='black', s=250, zorder=4, edgecolors='white', linewidths=1.5, label='Total Rank' if current_y == 0 else "")
        
        current_y += 1
    
    # Add year text label on the far right
    ax.text(26.5, current_y - 3, str(year), va='center', ha='left', fontsize=14, fontweight='bold', color='gray')
    
    # Add a visual separator line between years
    ax.axhline(current_y - 0.5, color='lightgray', linewidth=1, alpha=0.5)
    
    current_y += 1.0 # Space between year groups

# Formatting the axes
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=12)

ax.set_xticks(range(1, 27))
ax.set_xticklabels(range(1, 27), fontsize=10)
ax.set_xlabel('Rank', fontsize=14, fontweight='bold', labelpad=15)
ax.set_xlim(0, 26)

# Grid lines
ax.xaxis.grid(True, linestyle='-', color='lightgray', alpha=0.6, zorder=0)

# Clean up borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Legend at the top
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=14)

plt.title('Top 5 Countries Eurovision (2016-2021)', fontsize=26, fontweight='bold', pad=50)
plt.figtext(0.5, 0.92, 'How the public vote and the jury vote affect the total rank.', ha='center', fontsize=16, color='gray')

filepath = os.path.join(output_dir, "jury_vs_public_bias.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

print(f"Chart saved successfully to: {filepath}")
