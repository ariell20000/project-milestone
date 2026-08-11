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

print("Loading lyrics data for untranslated wordcloud...")
with open(os.path.join(script_dir, '..', 'eurovision-lyrics-2025.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

text_corpus = ""
winner_count = 0
for key, value in data.items():
    if value.get('Place') == '1':
        winner_count += 1
        lyrics = value.get('Lyrics', '')
        text_corpus += " " + str(lyrics)

# Basic cleaning
text_corpus = re.sub(r'\[.*?\]', '', text_corpus)
text_corpus = re.sub(r'\(.*?\)', '', text_corpus)
text_corpus = text_corpus.lower()

# Since this is multilingual, we can add some common multilingual stopwords or vocalizations
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "chorus", "verse", "bridge", "x2", "oh", "yeah", "la", "ah", 
    "ooh", "na", "da", "boom", "bam", "hey", "let", "et", "le", "la", "les", "de", "un", "une", "des",
    "en", "qui", "que", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "y", "a", "à"
])

print("Generating untranslated word cloud...")
wordcloud = WordCloud(
    width=1920, 
    height=1080, 
    background_color='#111111',
    colormap='cool', # Different colormap to distinguish from the translated one
    stopwords=custom_stopwords,
    max_words=150,
    collocations=False
).generate(text_corpus)

plt.figure(figsize=(24, 14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Eurovision Winning Songs (Native Languages)', fontsize=36, color='white', pad=30, fontweight='bold')

plt.gcf().set_facecolor('#111111')
plt.tight_layout(pad=0)

filepath = os.path.join(output_dir, "winning_lyrics_wordcloud_native.png")
plt.savefig(filepath, dpi=300, facecolor='#111111', bbox_inches='tight')
plt.close()

print(f"Untranslated wordcloud saved successfully to: {filepath}")
