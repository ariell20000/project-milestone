import json
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output", "wordcloud")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(script_dir, '..', 'eurovision-lyrics-2025.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["chorus", "verse", "bridge", "x2", "oh", "yeah", "la", "ah", "don", "can", "lei", "lai", "lee", "loo", "li", "lu", "cha", "ding", "dong", "pa", "ya", "yeh", "woo", "hoo", "di", "dum", "doo", "da", "ba", "ma", "ta", "ra", "ru", "rum", "pam", "ooh", "na", "boom", "bam", "hey", "let", "will", "now", "one", "know", "see", "come", "want", "go", "make", "take", "say", "got", "never", "can", "us", "time", "day", "way", "tell", "need", "feel", "think", "give", "look", "right", "nothing", "away", "gonna", "wanna", "much", "even", "every", "always", "still", "something", "everything", "around", "like", "well", "mine", "many", "ye", "whoa", "wa", "lala", "lalala", "diggi", "ley", "ho", "ha", "je", "te", "se", "ne"])

freq_winners = Counter()
freq_losers = Counter()

total_winners = 0
total_losers = 0

for key, value in data.items():
    place = str(value.get('Place', '')).strip()
    
    lyrics = value.get('Lyrics translation')
    if not lyrics or str(lyrics).strip() == "":
        lyrics = value.get('Lyrics', '')
    
    lyrics = str(lyrics).lower()
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    lyrics = re.sub(r'\(.*?\)', '', lyrics)
    
    # Distinct words per song
    words = re.findall(r'\b[a-z]{2,}\b', lyrics)
    unique_words = set(w for w in words if w not in custom_stopwords)
    
    if place == "1" or place.startswith("1st"):
        total_winners += 1
        for w in unique_words:
            freq_winners[w] += 1
    else:
        total_losers += 1
        for w in unique_words:
            freq_losers[w] += 1

# Convert to raw percentage (e.g. 0.10 for 10%)
perc_winners = {k: v / total_winners for k, v in freq_winners.items()} if total_winners > 0 else {}
perc_losers = {k: v / total_losers for k, v in freq_losers.items()} if total_losers > 0 else {}

print(f"Generating word clouds... Winners: {total_winners}, Losers: {total_losers}")

wc_winners = WordCloud(
    width=900, 
    height=1080, 
    background_color='#111111', 
    colormap='spring',
    max_words=100,
    relative_scaling=1
).generate_from_frequencies(perc_winners)

wc_losers = WordCloud(
    width=900, 
    height=1080, 
    background_color='#111111', 
    colormap='cool',
    max_words=100,
    relative_scaling=1
).generate_from_frequencies(perc_losers)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14), facecolor='#111111')

ax1.imshow(wc_winners, interpolation='bilinear')
ax1.axis('off')
ax1.set_title(f'Winners Only ({total_winners} Songs)', fontsize=28, color='white', pad=20, fontweight='bold')

ax2.imshow(wc_losers, interpolation='bilinear')
ax2.axis('off')
ax2.set_title(f'All Losers ({total_losers} Songs)', fontsize=28, color='white', pad=20, fontweight='bold')

plt.suptitle('Vocabulary: Winners vs. All Losers', fontsize=36, color='white', fontweight='bold', y=0.95)

explanation = "* Word size strictly reflects the percentage of songs in that group containing the word at least once."
plt.figtext(0.5, 0.03, explanation, ha='center', fontsize=22, color='lightgray', style='italic')

plt.tight_layout(rect=[0, 0.06, 1, 0.9])

filepath = os.path.join(output_dir, "winners_vs_losers_percentage.png")
plt.savefig(filepath, dpi=300, facecolor='#111111', bbox_inches='tight')
plt.close()

print(f"Wordcloud saved successfully to: {filepath}")
