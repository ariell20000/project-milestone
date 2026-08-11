from fpdf import FPDF
import os
import glob

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Eurovision Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, '..', 'output')
    output_pdf = os.path.join(script_dir, 'Eurovision_Analysis_Report.pdf')

    pdf = PDF()
    
    graphs = [
        {
            'file': 'composer_career_stats.png',
            'title': 'Composer Career Statistics',
            'desc': 'Scatter plot showing career total entries vs average final rank for Eurovision composers.'
        },
        {
            'file': 'composer_top5_finishes.png',
            'title': 'Top 5 Finishes by Composer',
            'desc': 'Bar chart showing the composers with the most top 5 finishes.'
        },
        {
            'file': 'friendship_clusters_all_time.png',
            'title': 'Voting Alliances',
            'desc': 'Network graph showing the strongest historical voting alliances between countries.'
        },
        {
            'file': 'kingmaker_scatter.png',
            'title': 'The Kingmakers',
            'desc': 'Scatter plot comparing the points a country receives against the points it gives out to winners.'
        },
        {
            'file': 'theme_clusters_all.png',
            'title': 'Themes of All Eurovision Songs',
            'desc': 'Circle packing layout showing dominant lyrical themes across all Eurovision songs.'
        },
        {
            'file': 'theme_clusters_winners.png',
            'title': 'Themes of Winning Songs',
            'desc': 'Circle packing layout highlighting the overall winning songs across themes.'
        },
        {
            'file': 'theme_clusters_jury_winners.png',
            'title': 'Jury Favorites (12 Points)',
            'desc': 'Circle packing layout highlighting songs that received 12 points from a jury.'
        },
        {
            'file': 'theme_clusters_public_winners.png',
            'title': 'Public Favorites (12 Points)',
            'desc': 'Circle packing layout highlighting songs that received 12 points from the public.'
        },
        {
            'file': 'winner_bias_timeline.png',
            'title': 'Winner Voting Bias (2016-2021)',
            'desc': 'Stacked bar chart showing the split of points from Jury vs. Public for winners.'
        },
        {
            'file': 'winners_by_language_pie.png',
            'title': 'Winners by Language',
            'desc': 'Pie chart showing the breakdown of winning songs by language.'
        },
        {
            'file': 'winning_lyrics_wordcloud.png',
            'title': 'Winning Lyrics (Translated)',
            'desc': 'Word cloud of translated winning Eurovision lyrics.'
        },
        {
            'file': 'winning_lyrics_wordcloud_native.png',
            'title': 'Winning Lyrics (Native)',
            'desc': 'Word cloud of native winning Eurovision lyrics.'
        },
        {
            'file': 'tastemaker_matrix.png',
            'title': 'The Eurovision Tastemaker Matrix',
            'desc': 'A matrix comparing how countries vote for winners versus losers.'
        },
        {
            'file': 'jury_vs_public_bias.png',
            'title': 'Jury vs Public Bias (Top 5)',
            'desc': 'How the public vote and the jury vote affect the total rank for the top 5 countries.'
        }
    ]

    for graph in graphs:
        img_path = os.path.join(images_dir, graph['file'])
        if os.path.exists(img_path):
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, graph['title'], 0, 1, 'L')
            
            # Description
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, graph['desc'])
            pdf.ln(5)
            
            # Image
            pdf.image(img_path, w=190)

    pdf.output(output_pdf, 'F')
    print(f"PDF generated successfully at: {output_pdf}")

if __name__ == "__main__":
    create_report()
