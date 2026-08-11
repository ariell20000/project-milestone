import json
import os
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading lyrics data for pie chart...")
with open(os.path.join(script_dir, '..', 'eurovision-lyrics-2025.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# Tally languages
language_counts = {}
for key, value in data.items():
    if value.get('Place') == '1':
        lang = value.get('Language', 'Unknown')
        
        # Take the primary language if multiple are listed
        lang = lang.split(',')[0].strip()
        
        # Consolidate spellings
        if "Crimean Tatar" in lang:
            lang = "Crimean Tatar"
            
        language_counts[lang] = language_counts.get(lang, 0) + 1

# Group small slices into 'Other'
THRESHOLD = 2
grouped_counts = {}
other_count = 0
for lang, count in language_counts.items():
    if count < THRESHOLD:
        other_count += count
    else:
        grouped_counts[lang] = count

if other_count > 0:
    grouped_counts['Other'] = other_count

# Sort languages by count (highest first)
sorted_langs = sorted(grouped_counts.items(), key=lambda x: x[1], reverse=True)

# Separate into labels and sizes
labels = [f"{k}\n{v}" for k, v in sorted_langs]
sizes = [v for k, v in sorted_langs]

# Use a vibrant, modern colormap built into matplotlib
colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(sizes)))

# Dark mode theme for a premium look
bg_color = '#1E1E2E'
text_color = '#FFFFFF'

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 14), facecolor=bg_color)
ax.set_facecolor(bg_color)

# Draw pie chart with white borders between slices for a cleaner look
wedges, texts = ax.pie(
    sizes, 
    labels=labels, 
    colors=colors, 
    startangle=140, # Rotated slightly for a better visual balance
    counterclock=False, 
    labeldistance=1.05,
    wedgeprops={'edgecolor': bg_color, 'linewidth': 3}, # Slice separation
    textprops={'fontsize': 14, 'fontweight': 'bold', 'color': text_color}
)

# Add title
plt.title('Eurovision Winners by Language', fontsize=42, fontweight='bold', color=text_color, pad=40)

# Add a subtle footer
plt.figtext(0.5, 0.05, "Data spanning 1956 - 2021", ha="center", fontsize=14, fontweight='bold', color='#A6ADC8')

filepath = os.path.join(output_dir, "winners_by_language_pie.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=bg_color)
plt.close()

print(f"Pie chart saved successfully to: {filepath}")
