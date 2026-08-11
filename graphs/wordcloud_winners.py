import json
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

# Create a folder called 'images' in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
print("Loading lyrics data...")
with open(os.path.join(script_dir, '..', 'eurovision-lyrics-2025.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# Collect lyrics for winning songs
text_corpus = ""
winner_count = 0
for key, value in data.items():
    if value.get('Place') == '1':
        winner_count += 1
        # Prefer English translation if available to make a unified word cloud
        lyrics = value.get('Lyrics translation')
        if not lyrics or str(lyrics).strip() == "":
            lyrics = value.get('Lyrics', '')
        
        text_corpus += " " + str(lyrics)

print(f"Found {winner_count} winning songs.")

# Basic cleaning: remove structural tags like [Chorus], (Verse 1), etc.
text_corpus = re.sub(r'\[.*?\]', '', text_corpus)
text_corpus = re.sub(r'\(.*?\)', '', text_corpus)
text_corpus = text_corpus.lower()

# Add some common Eurovision filler words to stopwords
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "chorus", "verse", "bridge", "x2", "oh", "yeah", "la", "ah", 
    "ooh", "na", "da", "boom", "bam", "hey", "let"
])

print("Generating word cloud...")
# Generate WordCloud
wordcloud = WordCloud(
    width=1920, 
    height=1080, 
    background_color='#111111', # Dark grey/black background
    colormap='spring',         # A bright, pop-music appropriate colormap (pink/yellow)
    stopwords=custom_stopwords,
    max_words=150,
    collocations=False         # Avoids duplicating phrases
).generate(text_corpus)

# Plotting
plt.figure(figsize=(24, 14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Eurovision Winning Songs (English Translated)', fontsize=36, color='white', pad=30, fontweight='bold')

# Set figure background to match wordcloud background
plt.gcf().set_facecolor('#111111')
plt.tight_layout(pad=0)

filepath = os.path.join(output_dir, "winning_lyrics_wordcloud.png")
plt.savefig(filepath, dpi=300, facecolor='#111111', bbox_inches='tight')
plt.close()

print(f"Wordcloud saved successfully to: {filepath}")
