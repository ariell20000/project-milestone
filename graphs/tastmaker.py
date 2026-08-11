import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a professional visual theme for the plots
sns.set_theme(style="whitegrid", context="talk")


def fetch_and_process_voting_data():
    """
    Reads the Kaggle eurovision_1957-2021.csv file.
    Calculates the actual final rankings mathematically from the points,
    then calculates the Oracle and Contrarian indices for every voting country.
    """
    print("1/4: Loading your local Kaggle dataset...")
    # Pointing directly to your local file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, '..', 'eurovision_1957-2021.csv'))

    # We only care about Final rounds (ignoring semi-finals if they exist in this dataset)
    # The 'Edition' column often contains 'semi-final'. Let's filter those out.
    df = df[~df['Edition'].str.contains('semi', case=False, na=False)]

    # Standardize country names (lowercase and strip spaces) so they match perfectly
    df['From'] = df['From'].str.lower().str.strip()
    df['To'] = df['To'].str.lower().str.strip()

    print("2/4: Calculating actual song placements (1st, 2nd, 3rd...) for every year...")
    # Group by Year and Receiving Country to sum up their total points
    yearly_totals = df.groupby(['Year', 'To'])['Points'].sum().reset_index()

    # Sort them to find out who won each year
    yearly_totals['Final_Rank'] = yearly_totals.groupby('Year')['Points'].rank(method='min', ascending=False)

    # Drop the total points, we just need the rank now
    rankings = yearly_totals[['Year', 'To', 'Final_Rank']]

    print("3/4: Merging actual placements back into individual voting records...")
    # Merge the final rank back onto the original voting matrix
    df_merged = pd.merge(df, rankings, on=['Year', 'To'])

    print("4/4: Calculating behavioral metrics for each country...")
    # 1. Did they vote for the winner? (Gave 10 or 12 points to the song that finished 1st)
    df_merged['Voted_For_Winner'] = (df_merged['Final_Rank'] == 1) & (df_merged['Points'] >= 10)

    # 2. Did they vote for a loser? (Gave 8, 10, or 12 points to a song that finished 21st or lower)
    df_merged['Voted_For_Loser'] = (df_merged['Final_Rank'] >= 21) & (df_merged['Points'] >= 8)

    # Group by the Voting Country ('From') to aggregate their historical behavior
    # We group by Year first to see if they triggered the flag in that specific year, then sum across all years
    yearly_behavior = df_merged.groupby(['From', 'Year']).agg(
        Hit_Winner=('Voted_For_Winner', 'max'),
        Hit_Loser=('Voted_For_Loser', 'max')
    ).reset_index()

    # Calculate final stats per country
    country_stats = yearly_behavior.groupby('From').agg(
        Total_Years_Voted=('Year', 'count'),
        Oracle_Count=('Hit_Winner', 'sum'),
        Contrarian_Count=('Hit_Loser', 'sum')
    ).reset_index()

    # Convert to Percentages (Index Score)
    country_stats['Oracle_Index'] = (country_stats['Oracle_Count'] / country_stats['Total_Years_Voted']) * 100
    country_stats['Contrarian_Index'] = (country_stats['Contrarian_Count'] / country_stats['Total_Years_Voted']) * 100

    # Filter out countries that haven't competed much to remove statistical noise
    country_stats = country_stats[country_stats['Total_Years_Voted'] >= 15]

    # Capitalize country names for the final plot
    country_stats['From'] = country_stats['From'].str.title()

    return country_stats


print("Initializing Oracle Index pipeline...")
df_stats = fetch_and_process_voting_data()

print("Drawing the quadrant matrix...")
plt.figure(figsize=(14, 10))

# Create the scatter plot
ax = sns.scatterplot(
    data=df_stats,
    x='Oracle_Index',
    y='Contrarian_Index',
    size='Total_Years_Voted',
    sizes=(50, 400),
    color='#4C72B0',
    edgecolor='black',
    alpha=0.7,
    legend=False
)

# Calculate the medians to draw the crosshairs (Quadrants)
x_median = df_stats['Oracle_Index'].median()
y_median = df_stats['Contrarian_Index'].median()

# Draw the quadrant dividing lines
plt.axvline(x_median, color='red', linestyle='--', alpha=0.5)
plt.axhline(y_median, color='red', linestyle='--', alpha=0.5)

# Annotate every single country so the audience can find their favorites
for i, row in df_stats.iterrows():
    plt.annotate(
        row['From'],
        (row['Oracle_Index'], row['Contrarian_Index']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        alpha=0.8
    )

# Add Quadrant Labels (Background Text)
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')

plt.text(df_stats['Oracle_Index'].max() * 0.95, df_stats['Contrarian_Index'].max() * 0.95,
         "THE WILDCARDS\n(Love the Winners AND the Losers)",
         fontsize=14, color='darkred', fontweight='bold', ha='right', bbox=props)

plt.text(df_stats['Oracle_Index'].min() * 1.05, df_stats['Contrarian_Index'].max() * 0.95,
         "THE CONTRARIANS / ECHO CHAMBERS\n(Ignore Winners, Vote for Neighbors)",
         fontsize=14, color='darkred', fontweight='bold', ha='left', bbox=props)

plt.text(df_stats['Oracle_Index'].max() * 0.95, df_stats['Contrarian_Index'].min() * 1.05,
         "THE ORACLES\n(Perfect Taste, Predict the Winners)",
         fontsize=14, color='darkgreen', fontweight='bold', ha='right', bbox=props)

plt.text(df_stats['Oracle_Index'].min() * 1.05, df_stats['Contrarian_Index'].min() * 1.05,
         "THE CENTRISTS\n(Vote for safe, middle-of-the-pack songs)",
         fontsize=14, color='gray', fontweight='bold', ha='left', bbox=props)

# Final formatting
# Ensure the title is not cut off
plt.gcf().subplots_adjust(top=0.88)
plt.title('The Eurovision Tastemaker Matrix:\nWhich countries actually dictate European music tastes?', fontsize=18,
          fontweight='bold', pad=20)
plt.xlabel('Oracle Score (% of years they gave Top Points to the Winner)', fontsize=14, fontweight='bold')
plt.ylabel('Contrarian Score (% of years they gave High Points to the Bottom 5)', fontsize=14, fontweight='bold')

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

filepath = os.path.join(output_dir, "tastemaker_matrix.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

print(f"Analysis complete! Chart saved to {filepath}")