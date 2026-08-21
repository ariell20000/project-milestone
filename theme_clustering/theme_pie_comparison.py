import pandas as pd
import matplotlib.pyplot as plt
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
theme_preds_path = os.path.join(script_dir, 'bert model', 'theme_predictions.csv')
output_dir = os.path.join(script_dir, "..", "graph_outputs")

os.makedirs(output_dir, exist_ok=True)

THEMES = [
    "love",
    "empowerment",
    "party / celebration",
    "hope",
    "nature",
    "heartbreak",
    "nostalgia",
    "spirituality",
    "loneliness",
    "war or conflict",
]

def get_random_color(seed_str):
    random.seed(seed_str)
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    return f"#{r:02x}{g:02x}{b:02x}"

THEME_COLORS = {theme: get_random_color(theme) for theme in THEMES}

def main():
    print("Loading data...")
    merged_df = pd.read_csv(theme_preds_path)
    
    df_winners = merged_df[merged_df['IsWinner'] == 1]
    df_losers = merged_df[merged_df['IsWinner'] == 0]
    
    print("Calculating theme distributions...")
    counts_all = merged_df['Dominant_Theme'].value_counts()
    counts_losers = df_losers['Dominant_Theme'].value_counts()
    counts_winners = df_winners['Dominant_Theme'].value_counts()
    
    for theme in THEMES:
        if theme not in counts_all: counts_all[theme] = 0
        if theme not in counts_losers: counts_losers[theme] = 0
        if theme not in counts_winners: counts_winners[theme] = 0
        
    sorted_themes = counts_all.sort_values(ascending=False).index
    
    counts_all = counts_all.reindex(sorted_themes)
    counts_losers = counts_losers.reindex(sorted_themes)
    counts_winners = counts_winners.reindex(sorted_themes)
    
    counts_all = counts_all[counts_all > 0]
    counts_losers = counts_losers[counts_losers > 0]
    counts_winners = counts_winners[counts_winners > 0]
    
    print("Plotting pie charts in 2-row layout...")
    fig = plt.figure(figsize=(18, 18), facecolor='#FAFAFA')
    fig.suptitle('Theme Distribution: All Songs vs. Losers vs. Winners', fontsize=32, fontweight='bold', color='#2C3E50', y=0.96)
    
    text_props = {'fontsize': 16, 'weight': 'bold'}
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.pie(counts_all, labels=[t.title() for t in counts_all.index], autopct='%1.1f%%', startangle=140, 
            colors=[THEME_COLORS[t] for t in counts_all.index], wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, textprops=text_props)
    ax1.set_title(f'All Songs ({len(merged_df)})', fontsize=26, fontweight='bold', color='#2C3E50', pad=20)
    
    ax2 = plt.subplot(2, 2, 3)
    ax2.pie(counts_losers, labels=[t.title() for t in counts_losers.index], autopct='%1.1f%%', startangle=140, 
            colors=[THEME_COLORS[t] for t in counts_losers.index], wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, textprops=text_props)
    ax2.set_title(f'Losers ({len(df_losers)})', fontsize=26, fontweight='bold', color='#2C3E50', pad=20)
    
    ax3 = plt.subplot(2, 2, 4)
    ax3.pie(counts_winners, labels=[t.title() for t in counts_winners.index], autopct='%1.1f%%', startangle=140, 
            colors=[THEME_COLORS[t] for t in counts_winners.index], wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, textprops=text_props)
    ax3.set_title(f'Winners ({len(df_winners)})', fontsize=26, fontweight='bold', color='#2C3E50', pad=20)
    
    plt.figtext(0.5, 0.03, "Each song is represented by its single most dominant NMF theme.", ha="center", fontsize=20, color='#555555', style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    filename = "theme_pie_comparison.png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    main()
