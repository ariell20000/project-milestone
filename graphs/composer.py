import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Added for string splitting
import os

# Set up image output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, '..', 'output')
os.makedirs(image_dir, exist_ok=True)

# Set a professional visual theme for the plots
sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------
# STEP 1: FETCH REAL DATA PIPELINE (API + CSV)
# ---------------------------------------------------------
def fetch_real_eurovision_data():
    """
    Fetches official contest rankings and pre-scraped Wikipedia composer data
    directly from the Spijkervet/Plotly open dataset.
    """
    print("1/3: Fetching official dataset from GitHub (Plotly/Spijkervet mirror)...")
    rankings_url = "https://raw.githubusercontent.com/plotly/Figure-Friday/main/2024/week-40/contestants.csv"
    df = pd.read_csv(rankings_url)

    # The CSV actually has a built-in 'composers' column!
    # This guarantees 100% coverage and avoids the sparse Wikidata API.
    df = df[['year', 'to_country', 'place_final', 'composers']].dropna()
    df.rename(columns={'to_country': 'Country', 'year': 'Year', 'place_final': 'Final_Rank'}, inplace=True)

    print("2/3: Cleaning and expanding songwriting teams...")
    # Composers are usually comma-separated strings (e.g., "Thomas G:son, Peter Boström")
    # We standardize the separators (including semicolons) and split them into lists
    df['composers'] = df['composers'].str.replace(r' and |&|\n|;', ',', regex=True)
    df['Composer'] = df['composers'].str.split(',')

    print("3/3: Exploding lists into individual rows...")
    # Create one row per composer per song
    df = df.explode('Composer')
    df['Composer'] = df['Composer'].str.strip()

    # Filter out empty strings or unknown
    df = df[df['Composer'].str.len() > 1]
    df = df[~df['Composer'].str.contains('Unknown', case=False, na=False)]

    return df


print("Initializing live data pipeline (this may take a few seconds)...")
df = fetch_real_eurovision_data()

# ---------------------------------------------------------
# STEP 2: AGGREGATE STATS PER COMPOSER
# ---------------------------------------------------------
print("Calculating hit rates and track records...")

# Group by Composer to get their career stats
composer_stats = df.groupby('Composer').agg(
    Total_Entries=('Final_Rank', 'count'),
    Avg_Rank=('Final_Rank', 'mean'),
    Top_5_Finishes=('Final_Rank', lambda x: (x <= 5).sum()),
    Wins=('Final_Rank', lambda x: (x == 1).sum())
).reset_index()

# Calculate "Clutch Rate" (% of their songs that hit Top 5)
composer_stats['Top_5_Rate'] = (composer_stats['Top_5_Finishes'] / composer_stats['Total_Entries']) * 100

# Filter out the one-hit wonders for the label generation
# Raised to >= 5 because we now have the FULL dataset of thousands of composers!
veterans = composer_stats[composer_stats['Total_Entries'] >= 5]

# Add a tiny bit of random jitter to the Total_Entries so bubbles don't perfectly overlap
np.random.seed(42)  # For reproducibility
composer_stats['Jittered_Entries'] = composer_stats['Total_Entries'] + np.random.uniform(-0.15, 0.15,
                                                                                         size=len(composer_stats))

# ---------------------------------------------------------
# STEP 3: BUILD THE STORY-DRIVEN DASHBOARD
# ---------------------------------------------------------
print("Drawing the charts...")

# --- PLOT 1: The Landscape (Bubble Chart) ---
plt.figure(figsize=(14, 8))
plt.suptitle('Eurovision Hitmakers: Average Placement vs. Total Songs Written', fontsize=18, fontweight='bold')

sns.scatterplot(
    data=composer_stats,
    x='Jittered_Entries',  # Use jittered data for readability
    y='Avg_Rank',
    size='Top_5_Finishes',
    sizes=(50, 800),  # Bubble size range
    hue='Total_Entries',
    palette='viridis',
    alpha=0.6,
    edgecolor='black',
    legend=False
)

# Invert Y-axis so Rank 1 (Best) is at the top
plt.gca().invert_yaxis()
plt.xlabel('Total Eurovision Entries Written (with slight visual jitter)', fontsize=14, fontweight='bold')
plt.ylabel('Average Final Rank (1st = Best)', fontsize=14, fontweight='bold')
plt.yticks(np.arange(1, 27, 2))

# Dynamically find the top 30 most prolific composers to annotate on the scatter plot
top_prolific = veterans.nlargest(30, 'Total_Entries')

for i, row in top_prolific.iterrows():
    plt.annotate(
        str(row['Composer']),
        (row['Total_Entries'], row['Avg_Rank']),
        xytext=(10, 5), textcoords='offset points',
        fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0)
    )

# Add a threshold line for a "Top 10 Average"
plt.axhline(10, color='red', linestyle='--', alpha=0.5, label='Top 10 Average')
plt.text(x=max(composer_stats['Total_Entries']) - 0.5, y=9.5, s="Top 10 Tier", color='red', alpha=0.7,
         fontweight='bold')

plt.tight_layout()
out_path1 = os.path.join(image_dir, 'composer_career_stats.png')
plt.savefig(out_path1, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_path1}")
plt.close()

# --- PLOT 2: The "Clutch" Performers (Bar Chart) ---
plt.figure(figsize=(12, 12))
plt.suptitle('The Trophy Cabinet: Total Top 5 Finishes', fontsize=18, fontweight='bold')

# Dynamically select the Top 30 composers based purely on total Top 5 finishes
top_hitmakers = veterans.nlargest(30, 'Top_5_Finishes')

ax2 = sns.barplot(
    data=top_hitmakers,
    x='Top_5_Finishes',
    y='Composer',
    hue='Composer',  # Fixes the Seaborn palette warning!
    palette='magma',
    legend=False,
    edgecolor='black'
)

plt.xlabel('Number of Top 5 Finishes', fontsize=14, fontweight='bold')
plt.ylabel('')

# Add the absolute count numbers on the bars
for p in ax2.patches:
    width = p.get_width()
    plt.annotate(f"{int(width)}",
                 (width, p.get_y() + p.get_height() / 2.),
                 ha='left', va='center',
                 fontsize=10, fontweight='bold', color='black', xytext=(5, 0),
                 textcoords='offset points')

plt.tight_layout()
out_path2 = os.path.join(image_dir, 'composer_top5_finishes.png')
plt.savefig(out_path2, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_path2}")
plt.close()

print("Analysis complete!")