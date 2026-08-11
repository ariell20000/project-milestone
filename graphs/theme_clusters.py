import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import circlify
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'model', 'theme_predictions.csv')
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define features and their colors to match a nice aesthetic
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

# We assign an aesthetically pleasing color palette for the 10 themes
COLORS = {
    "love": "#CD5C5C",             # Indian Red
    "empowerment": "#5D9ECE",      # Blue
    "nature": "#66B27C",           # Green
    "hope": "#F4D03F",             # Yellow
    "party / celebration": "#E67E22", # Orange
    "loneliness": "#9B59B6",       # Purple
    "nostalgia": "#95A5A6",        # Grey
    "spirituality": "#76D7C4",     # Teal
    "heartbreak": "#F5B7B1",       # Light Pink
    "war or conflict": "#34495E"   # Midnight Blue
}

def pack_circles_fermat(n, r_circle=1.0):
    """Generates x, y coordinates for n circles using Fermat's spiral (sunflower layout)."""
    points = []
    # c determines the spacing. For circles of radius r, c around 1.5*r works well.
    c = 1.6 * r_circle
    for i in range(1, n + 1):
        # Golden angle
        theta = i * 2.39996322972865332
        r = c * math.sqrt(i)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append((x, y))
    return points

def create_bubble_chart(df, title, filename, highlight_col=None):
    # Filter the dataframe so we ONLY draw the songs we care about!
    if highlight_col:
        plot_df = df[df[highlight_col] == 1].copy()
    else:
        plot_df = df.copy()

    # Set up figure
    fig, axes = plt.subplots(2, 5, figsize=(22, 12), facecolor='#FAFAFA')
    fig.suptitle(title, fontsize=28, fontweight='bold', color='#2C3E50', y=0.95)
    
    axes = axes.flatten()
    
    counts = plot_df['Dominant_Theme'].value_counts()
    max_count = counts.max() if not counts.empty else 1
    # Max radius of the whole cluster would be c * sqrt(max_count). 
    # We use this to set consistent axis limits for all subplots so larger clusters look larger!
    c_spacing = 1.6
    r_circle = 1.0
    global_max_radius = c_spacing * math.sqrt(max_count) + r_circle + 2.0
    
    for idx, theme in enumerate(THEMES):
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')
        ax.axis('off')
        ax.set_aspect('equal')
        
        theme_songs = plot_df[plot_df['Dominant_Theme'] == theme].copy()
        count = len(theme_songs)
        
        if count == 0:
            # Add empty label
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=COLORS[theme], lw=1.5)
            ax.text(0, -global_max_radius * 0.95, f"{theme.capitalize()}\n(0)", ha="center", va="center", 
                    fontsize=12, fontweight='bold', color=COLORS[theme], bbox=bbox_props)
            ax.set_xlim(-global_max_radius, global_max_radius)
            ax.set_ylim(-global_max_radius, global_max_radius)
            continue
            
        # Get coordinates for tight packing
        coords = pack_circles_fermat(count, r_circle=r_circle)
        
        ax.set_xlim(-global_max_radius, global_max_radius)
        ax.set_ylim(-global_max_radius, global_max_radius)
        
        # Plot circles
        for i, (_, row) in enumerate(theme_songs.iterrows()):
            x, y = coords[i]
            
            # Since we filtered the dataframe, ALL drawn circles belong to the category!
            # We give them all a solid black border.
            edge_color = 'black'
            line_width = 1.5
            alpha = 1.0
            
            # Except if we are plotting ALL songs (highlight_col is None), then no heavy border.
            if highlight_col is None:
                edge_color = COLORS[theme]
                line_width = 0.5
                alpha = 0.85
            
            c = patches.Circle((x, y), r_circle, alpha=alpha, 
                               facecolor=COLORS[theme], edgecolor=edge_color, lw=line_width)
            ax.add_patch(c)
            
        # Add label at the bottom
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=COLORS[theme], lw=1.5)
        ax.text(0, -global_max_radius * 0.95, f"{theme.capitalize()}", ha="center", va="top", 
                fontsize=13, fontweight='bold', color=COLORS[theme], bbox=bbox_props)
                
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.3)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"Saved {filename}")
    plt.close()

def main():
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please run generate_themes.py first.")
        return
        
    print(f"Loading predictions from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Generating All Songs chart...")
    create_bubble_chart(df, "Dominant Themes in Eurovision Lyrics (All Songs)", "theme_clusters_all.png", highlight_col=None)
    
    print("Generating Winners chart...")
    create_bubble_chart(df, "Dominant Themes in Eurovision Lyrics (Overall Winners)", "theme_clusters_winners.png", highlight_col='IsWinner')
    
    print("Generating Jury Winners chart...")
    create_bubble_chart(df, "What the Jury Loves: Jury 1st Place Nominations", "theme_clusters_jury_winners.png", highlight_col='IsJury12')
    
    print("Generating Public Winners chart...")
    create_bubble_chart(df, "What the Public Loves: Public 1st Place Nominations", "theme_clusters_public_winners.png", highlight_col='IsPublic12')

if __name__ == "__main__":
    main()
