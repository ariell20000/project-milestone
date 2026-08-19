import pandas as pd
import matplotlib.pyplot as plt
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'test_lang', 'lyrics_nmf_topics_standalone', 'eurovision_enriched1.csv')
theme_preds_path = os.path.join(script_dir, '..', 'model', 'theme_predictions.csv')
output_dir = os.path.join(script_dir, "..", "output")

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

def get_dominant_theme_counts(df):
    plot_df = df.copy()
    for theme in THEMES:
        plot_df[theme] = pd.to_numeric(plot_df[theme], errors='coerce').fillna(0)
        
    theme_scores = plot_df[THEMES]
    max_themes = theme_scores.idxmax(axis=1)
    max_scores = theme_scores.max(axis=1)
    
    valid_mask = max_scores > 0
    dominant_themes = max_themes[valid_mask]
    return dominant_themes.value_counts()

def main():
    print("Loading data...")
    df = pd.read_csv(data_path)
    preds_df = pd.read_csv(theme_preds_path)
    
    preds_df['Year'] = preds_df['Year'].astype(str)
    df['Year'] = df['Year'].astype(str)
    
    merged_df = pd.merge(df, preds_df[['Song', 'Year', 'IsWinner']], 
                         on=['Song', 'Year'], how='left')
    merged_df['IsWinner'] = merged_df['IsWinner'].fillna(0)
    
    df_winners = merged_df[merged_df['IsWinner'] == 1]
    df_losers = merged_df[merged_df['IsWinner'] == 0]
    
    print("Calculating theme distributions...")
    counts_all = get_dominant_theme_counts(merged_df)
    counts_losers = get_dominant_theme_counts(df_losers)
    counts_winners = get_dominant_theme_counts(df_winners)
    
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
    
    print("Plotting side-by-side pie charts...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), facecolor='#FAFAFA')
    fig.suptitle('Theme Distribution: All Songs vs. Losers vs. Winners', fontsize=28, fontweight='bold', color='#2C3E50', y=1.05)
    
    ax1.pie(counts_all, labels=[t.title() for t in counts_all.index], autopct='%1.1f%%', startangle=140, 
            colors=[THEME_COLORS[t] for t in counts_all.index], wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax1.set_title(f'All Songs ({len(merged_df)})', fontsize=20, fontweight='bold', color='#2C3E50', pad=20)
    
    ax2.pie(counts_losers, labels=[t.title() for t in counts_losers.index], autopct='%1.1f%%', startangle=140, 
            colors=[THEME_COLORS[t] for t in counts_losers.index], wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax2.set_title(f'Losers ({len(df_losers)})', fontsize=20, fontweight='bold', color='#2C3E50', pad=20)
    
    ax3.pie(counts_winners, labels=[t.title() for t in counts_winners.index], autopct='%1.1f%%', startangle=140, 
            colors=[THEME_COLORS[t] for t in counts_winners.index], wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax3.set_title(f'Winners ({len(df_winners)})', fontsize=20, fontweight='bold', color='#2C3E50', pad=20)
    
    plt.figtext(0.5, 0.05, "Each song is represented by its single most dominant NMF theme.", ha="center", fontsize=16, color='#555555', style='italic')
    
    plt.tight_layout()
    
    filename = "theme_pie_comparison.png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    main()
