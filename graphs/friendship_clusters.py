import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data from parent directory
df = pd.read_csv(os.path.join(script_dir, '..', 'eurovision_1957-2021.csv'))

# Clean columns
df.columns = df.columns.str.strip()
df['Points type'] = df['Points type'].str.strip()
df['From'] = df['From'].str.strip().str.title()
df['To'] = df['To'].str.strip().str.title()

def standardize_name(name):
    # Fix inconsistencies in the original data before grouping
    fixes = {
        "United-Kingdom": "United Kingdom",
        "Bosnia-&-Herzegovina": "Bosnia & Herzegovina",
        "The-Netherlands": "The Netherlands",
        "North-Macedonia": "North Macedonia",
        "Czech-Republic": "Czech Republic",
        "San-Marino": "San Marino",
        "Serbia-&-Montenegro": "Serbia & Montenegro",
    }
    return fixes.get(name, name)

df['From'] = df['From'].apply(standardize_name)
df['To'] = df['To'].apply(standardize_name)

# We only care about data from 1975 onwards
df = df[df['Year'] >= 1975].copy()
df_yearly = df.groupby(['Year', 'From', 'To'])['Points'].sum().reset_index()

print("Calculating all-time stats...")
stats = df_yearly.groupby(['From', 'To']).agg(
    Avg_Points=('Points', 'mean'),
    Votes_Cast=('Points', 'count')
).reset_index()

stats = stats[stats['Votes_Cast'] >= 5]
best_friends = stats.loc[stats.groupby('From')['Avg_Points'].idxmax()].copy()

G = nx.DiGraph()
for _, row in best_friends.iterrows():
    G.add_edge(row['From'], row['To'], weight=row['Avg_Points'])

G.remove_nodes_from(list(nx.isolates(G)))

# Square figure — prevents the vertical stretching issue
fig, ax = plt.subplots(figsize=(28, 28), facecolor='white')
ax.set_facecolor('white')

# spring_layout with high iterations for stability
# Increased k significantly to prevent nodes from overlapping
pos = nx.spring_layout(G, k=15.0, scale=3.0, iterations=1200, seed=42)

# Draw edges first (underneath)
edges = G.edges(data=True)
weights = [d['weight'] for u, v, d in edges]

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("gor", ["green", "orange", "red"])
vmin, vmax = 4, 12

nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color=weights, edge_cmap=cmap,
    edge_vmin=vmin, edge_vmax=vmax,
    arrows=True, arrowsize=50, arrowstyle='-|>',
    connectionstyle="arc3,rad=0.12", width=2.5, alpha=0.9
)


# Helper to abbreviate names to 3-5 letters
def abbreviate(name):
    overrides = {
        "United Kingdom": "UK",
        "United-Kingdom": "UK",
        "Bosnia & Herzegovina": "B&H",
        "Bosnia-&-Herzegovina": "B&H",
        "The Netherlands": "NLD",
        "The-Netherlands": "NLD",
        "North Macedonia": "MKD",
        "North-Macedonia": "MKD",
        "Czech Republic": "CZE",
        "Czech-Republic": "CZE",
        "San Marino": "SMR",
        "San-Marino": "SMR",
        "Serbia & Montenegro": "S&M",
        "Serbia-&-Montenegro": "S&M",
        "Switzerland": "SUI",
        "Germany": "GER",
        "France": "FRA",
        "Spain": "ESP",
        "Italy": "ITA",
        "Portugal": "POR",
        "Sweden": "SWE",
        "Norway": "NOR",
        "Denmark": "DEN",
        "Finland": "FIN",
        "Iceland": "ISL",
        "Ireland": "IRE",
        "Belgium": "BEL",
        "Austria": "AUT",
        "Greece": "GRE",
        "Cyprus": "CYP",
        "Turkey": "TUR",
        "Israel": "ISR",
        "Russia": "RUS",
        "Ukraine": "UKR",
        "Belarus": "BLR",
        "Poland": "POL",
        "Croatia": "CRO",
        "Slovenia": "SLO",
        "Hungary": "HUN",
        "Romania": "ROU",
        "Estonia": "EST",
        "Latvia": "LAT",
        "Lithuania": "LIT",
        "Moldova": "MDA",
        "Armenia": "ARM",
        "Azerbaijan": "AZE",
        "Georgia": "GEO",
        "Albania": "ALB",
        "Malta": "MLT",
        "Serbia": "SRB",
        "Bulgaria": "BUL",
        "Australia": "AUS",
        "Yugoslavia": "YUG",
        "Monaco": "MON",
        "Luxembourg": "LUX"
    }
    if name in overrides:
        return overrides[name]
    if len(name) <= 5:
        return name.upper()
    return name[:4].upper()


# Draw nodes — smaller size since text is shorter
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#d6eaf8', node_size=3200,
                       alpha=1.0, edgecolors='#2c3e50', linewidths=2.0)

# Draw node labels inside nodes with abbreviations
labels = {node: abbreviate(node) for node in G.nodes()}
nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=24, font_family='sans-serif',
                        font_weight='bold', font_color='#1a1a2e')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30, shrink=0.6)
cbar.set_label('Average Points Given', fontsize=26, fontweight='bold', rotation=270, labelpad=40)
cbar.ax.tick_params(labelsize=20)

# Title
ax.set_title('Eurovision Friendship Clusters (All-Time: 1975-2021)', fontsize=36, fontweight='bold', pad=50)

# Explanation text (moved to top-left to avoid overlap)
explanation_text = (
    "Each arrow: Country A → Country B (B received highest avg points from A).\n"
    "Colors: Green → Orange → Red (weaker to stronger connection)."
)
ax.text(0.02, 0.98, explanation_text, ha='left', va='top', transform=ax.transAxes,
        fontsize=22, style='italic', bbox=dict(facecolor='whitesmoke', alpha=0.9, boxstyle='round,pad=0.8'))

ax.axis('off')
plt.tight_layout()

filepath = os.path.join(output_dir, "friendship_clusters_all_time.png")
plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"All-time image saved successfully to: {filepath}")
