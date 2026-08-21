import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', '..', 'eurovision_1957-2021.csv')
theme_path = os.path.join(script_dir, 'theme_predictions.csv')

print("Loading data...")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Clean point type
df['Points type'] = df['Points type'].str.strip().str.lower()
def map_point_type(pt):
    if 'jury' in pt:
        return 'Jury'
    elif 'televote' in pt:
        return 'Public'
    return 'Other'

df['Type'] = df['Points type'].apply(map_point_type)
df['To'] = df['To'].str.strip().str.title()
# Fix specific country names if needed
df['To'] = df['To'].replace({'Bosnia & Herzegovina': 'Bosnia and Herzegovina', 'North Macedonia': 'North Macedonia'})

# Get songs that received 12 points (1st place nomination) from Jury or Public
jury_12s = df[(df['Type'] == 'Jury') & (df['Points'] == 12)][['Year', 'To']].drop_duplicates().copy()
jury_12s['IsJury12'] = 1

public_12s = df[(df['Type'] == 'Public') & (df['Points'] == 12)][['Year', 'To']].drop_duplicates().copy()
public_12s['IsPublic12'] = 1

# Load theme predictions
themes = pd.read_csv(theme_path)

# Ensure previous run columns are dropped if they exist, so we don't duplicate
cols_to_drop = [c for c in ['IsJuryWinner', 'IsPublicWinner', 'IsJury12', 'IsPublic12'] if c in themes.columns]
themes = themes.drop(columns=cols_to_drop)

# Merge
themes['Country'] = themes['Country'].str.strip().str.title()
themes = pd.merge(themes, jury_12s, left_on=['Year', 'Country'], right_on=['Year', 'To'], how='left').drop(columns=['To'])
themes = pd.merge(themes, public_12s, left_on=['Year', 'Country'], right_on=['Year', 'To'], how='left').drop(columns=['To'])

themes['IsJury12'] = themes['IsJury12'].fillna(0).astype(int)
themes['IsPublic12'] = themes['IsPublic12'].fillna(0).astype(int)

# Check if we successfully mapped them
print(f"Songs with Jury 12s found: {themes['IsJury12'].sum()}")
print(f"Songs with Public 12s found: {themes['IsPublic12'].sum()}")

# Overwrite
themes.to_csv(theme_path, index=False)
print("Updated theme_predictions.csv with IsJury12 and IsPublic12.")
