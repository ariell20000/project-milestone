import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'test_lang', 'lyrics_nmf_topics_standalone', 'eurovision_enriched1.csv')
theme_preds_path = os.path.join(script_dir, '..', 'model', 'theme_predictions.csv')
output_dir = os.path.join(script_dir, "..", "output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_random_color(seed_str):
    random.seed(seed_str)
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def pack_circles_fermat(n, r_circle=1.0):
    points = []
    c = 1.6 * r_circle
    for i in range(1, n + 1):
        theta = i * 2.39996322972865332
        r = c * math.sqrt(i)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))
    return points

def create_bubble_chart(df, title, filename, highlight_col=None):
    if highlight_col:
        plot_df = df[df[highlight_col] == 1].copy()
    else:
        plot_df = df.copy()

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

    n_themes = len(THEMES)
    cols = 5
    rows = math.ceil(n_themes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(22, 5 * rows), facecolor='#FAFAFA')
    fig.suptitle(title, fontsize=28, fontweight='bold', color='#2C3E50', y=0.95 + (0.01 * rows))
    
    if n_themes > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    max_count = 1
    THRESHOLD = 0.8
    for theme in THEMES:
        plot_df[theme] = pd.to_numeric(plot_df[theme], errors='coerce').fillna(0)
        count = len(plot_df[plot_df[theme] >= THRESHOLD])
        if count > max_count:
            max_count = count
            
    c_spacing = 1.6
    r_circle = 1.0
    global_max_radius = c_spacing * math.sqrt(max_count) + r_circle + 2.0
    
    for idx, theme in enumerate(THEMES):
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')
        ax.axis('off')
        ax.set_aspect('equal')
        
        theme_songs = plot_df[plot_df[theme] >= THRESHOLD].sort_values(by=theme, ascending=False)
        count = len(theme_songs)
        color = get_random_color(theme)
        
        if count == 0:
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1.5)
            ax.text(0, -global_max_radius * 0.95, f"{theme.capitalize()}\n(0)", ha="center", va="center", 
                    fontsize=12, fontweight='bold', color=color, bbox=bbox_props)
            ax.set_xlim(-global_max_radius, global_max_radius)
            ax.set_ylim(-global_max_radius, global_max_radius)
            continue
            
        coords = pack_circles_fermat(count, r_circle=r_circle)
        
        ax.set_xlim(-global_max_radius, global_max_radius)
        ax.set_ylim(-global_max_radius, global_max_radius)
        
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        
        scores = theme_songs[theme].values
        sizes = 80
        
        edge_color = 'black' if highlight_col else color
        line_width = 1.5 if highlight_col else 0.5
        alpha = 1.0 if highlight_col else 0.85
        
        ax.scatter(x_vals, y_vals, s=sizes, c=color, edgecolors=edge_color, linewidths=line_width, alpha=alpha)
            
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1.5)
        ax.text(0, -global_max_radius * 0.95, f"{theme.title()}", ha="center", va="top", 
                fontsize=13, fontweight='bold', color=color, bbox=bbox_props)
                
    for i in range(len(THEMES), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Saved {filename}")
    plt.close()

def create_pie_chart(df, title, filename, highlight_col=None):
    if highlight_col:
        plot_df = df[df[highlight_col] == 1].copy()
    else:
        plot_df = df.copy()

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

    for theme in THEMES:
        plot_df[theme] = pd.to_numeric(plot_df[theme], errors='coerce').fillna(0)
    
    # Assign exactly one theme per song (the one with the highest score)
    theme_scores = plot_df[THEMES]
    max_themes = theme_scores.idxmax(axis=1)
    max_scores = theme_scores.max(axis=1)
    
    # Only include songs that have a non-zero max score
    valid_mask = max_scores > 0
    dominant_themes = max_themes[valid_mask]
    
    theme_counts = dominant_themes.value_counts()
    
    plt.figure(figsize=(10, 8), facecolor='#FAFAFA')
    plt.title(title, fontsize=20, fontweight='bold', color='#2C3E50', pad=20)
    plt.figtext(0.5, 0.05, "Each song is represented by its single most dominant theme.", ha="center", fontsize=12, color='#555555', style='italic')
    
    colors = [get_random_color(theme) for theme in theme_counts.index]
    
    plt.pie(theme_counts, labels=[t.title() for t in theme_counts.index], 
            autopct='%1.1f%%', startangle=140, colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Saved {filename}")
    plt.close()

def main():
    print("Loading data...")
    df = pd.read_csv(data_path)
    preds_df = pd.read_csv(theme_preds_path)
    
    preds_df['Year'] = preds_df['Year'].astype(str)
    df['Year'] = df['Year'].astype(str)
    
    merged_df = pd.merge(df, preds_df[['Song', 'Year', 'IsWinner']], 
                         on=['Song', 'Year'], how='left')
                         
    merged_df['IsWinner'] = merged_df['IsWinner'].fillna(0)
    
    print("Generating All Songs chart...")
    create_bubble_chart(merged_df, "Dominant Themes in Eurovision Lyrics (All Songs)", "theme_clusters_all.png", highlight_col=None)
    create_pie_chart(merged_df, "Theme Distribution (All Songs)", "theme_pie_all.png", highlight_col=None)
    
    print("Generating Winners chart...")
    create_bubble_chart(merged_df, "Dominant Themes in Eurovision Lyrics (Overall Winners)", "theme_clusters_winners.png", highlight_col='IsWinner')
    create_pie_chart(merged_df, "Theme Distribution (Overall Winners)", "theme_pie_winners.png", highlight_col='IsWinner')

if __name__ == "__main__":
    main()





