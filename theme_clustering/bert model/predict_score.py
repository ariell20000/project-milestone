import torch
from transformers import pipeline
import pandas as pd
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Features to score
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
json_path = os.path.join(script_dir, '..', 'eurovision-lyrics-2025.json')

def load_data():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for k, v in data.items():
        score = v.get('Score')
        lyrics = v.get('Lyrics')
        
        # We need both lyrics and a numeric score
        if not lyrics or not score:
            continue
            
        try:
            score = float(score)
            rows.append({'Score': score, 'Lyrics': lyrics})
        except ValueError:
            # Score was '-' or something non-numeric
            continue
            
    df = pd.DataFrame(rows)
    return df

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} songs with valid scores and lyrics.")
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA (GPU)" if device == 0 else "CPU"
    print(f"\nInitializing BERT model on device: {device_name}")
    
    # Load zero-shot classifier
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=device,
    )
    
    print("\nExtracting semantic features from lyrics. This may take a while...")
    
    # We will score in batches using the generator approach
    def lyrics_gen():
        for text in df['Lyrics'].tolist():
            yield text if text.strip() else "no lyrics"
            
    BATCH_SIZE = 16
    result_iter = classifier(lyrics_gen(), candidate_labels=FEATURES, multi_label=True, batch_size=BATCH_SIZE)
    
    features_list = []
    for result in tqdm(result_iter, total=len(df)):
        # result['labels'] contains labels sorted by score, result['scores'] are the corresponding scores
        feature_dict = dict(zip(result['labels'], result['scores']))
        features_list.append(feature_dict)
        
    features_df = pd.DataFrame(features_list)
    
    # Combine original df with features
    final_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    print("\n--- Training Linear Regression Model ---")
    X = final_df[FEATURES]
    y = final_df['Score']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} songs, evaluating on {len(X_test)} songs.")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance (Test Set):")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    print("\nFeature Coefficients (How much each topic affects the score):")
    # Using the columns from X to match the coefficients since X[FEATURES] maintains the order
    coefs = pd.DataFrame({'Feature': FEATURES, 'Coefficient': model.coef_})
    coefs = coefs.sort_values(by='Coefficient', ascending=False)
    for _, row in coefs.iterrows():
        print(f"  {row['Feature']:<20}: {row['Coefficient']:.2f}")
        
if __name__ == "__main__":
    main()
