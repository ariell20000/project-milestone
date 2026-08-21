import torch
from transformers import pipeline
import pandas as pd
import json
import os
from tqdm.auto import tqdm

FEATURES = [
    "love",
    "hope",
    "heartbreak",
    "party / celebration",
    "war or conflict",
    "nostalgia",
    "empowerment",
    "loneliness",
    "nature",
    "spirituality",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, '..', '..', 'eurovision-lyrics-2025.json')
output_csv = os.path.join(script_dir, 'theme_predictions.csv')

def load_data():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for k, v in data.items():
        lyrics = v.get('Lyrics translation')
        if not lyrics or str(lyrics).strip() == "":
            lyrics = v.get('Lyrics')
            
        if not lyrics or str(lyrics).strip() == "":
            continue
            
        is_winner = 1 if v.get('Place') == '1' else 0
        
        rows.append({
            'SongId': k,
            'Year': v.get('Year'),
            'Country': v.get('Country'),
            'Song': v.get('Song'),
            'Lyrics': str(lyrics),
            'IsWinner': is_winner
        })
        
    return pd.DataFrame(rows)

def main():
    if os.path.exists(output_csv):
        print(f"{output_csv} already exists. Delete it to regenerate.")
        return

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} songs with lyrics.")
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA (GPU)" if device == 0 else "CPU"
    print(f"\nInitializing BERT model on device: {device_name}")
    
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=device
    )
    
    def lyrics_gen():
        for text in df['Lyrics'].tolist():
            # Truncate slightly to avoid max length errors if lyrics are huge
            yield text[:2500] if len(text) > 2500 else text
            
    BATCH_SIZE = 16
    print("Running zero-shot classification...")
    result_iter = classifier(lyrics_gen(), candidate_labels=FEATURES, multi_label=False, batch_size=BATCH_SIZE)
    
    dominant_themes = []
    for result in tqdm(result_iter, total=len(df)):
        # result['labels'][0] is the top label since they are sorted by score
        dominant_themes.append(result['labels'][0])
        
    df['Dominant_Theme'] = dominant_themes
    
    # Drop lyrics to save space in the CSV
    df = df.drop(columns=['Lyrics'])
    df.to_csv(output_csv, index=False)
    print(f"\nSaved dominant themes to {output_csv}")

if __name__ == "__main__":
    main()
