import matplotlib.pyplot as plt
import plotly.express as px
import os

def create_plots(df):
    os.makedirs("outputs/plots", exist_ok=True)

    # Matplotlib 1 - Histogram
    plt.figure()
    df['popularity'].hist()
    plt.title("Popularity Distribution")
    plt.xlabel("Popularity")
    plt.ylabel("Count")
    plt.savefig("outputs/plots/histogram.png")

    # Matplotlib 2 - Scatter
    plt.figure()
    plt.scatter(df['energy'], df['popularity'])
    plt.title("Energy vs Popularity")
    plt.xlabel("Energy")
    plt.ylabel("Popularity")
    plt.savefig("outputs/plots/scatter.png")

    # Matplotlib 3 - Bar
    if 'popular' not in df.columns:
        df = df.copy()
        df['popular'] = (df['popularity'] > 70).astype(int)

    plt.figure()
    df.groupby('popular')['energy'].mean().plot(kind='bar')
    plt.title("Average Energy by Popularity")
    plt.xlabel("Popular (0/1)")
    plt.ylabel("Energy")
    plt.savefig("outputs/plots/bar.png")

    # Plotly 1
    fig1 = px.scatter(df, x='danceability', y='popularity', title="Danceability vs Popularity")
    fig1.write_html("outputs/plots/plotly_scatter.html")

    # Plotly 2
    numeric_df = df.select_dtypes(include="number")
    fig2 = px.imshow(numeric_df.corr(), title="Correlation Matrix")
    fig2.write_html("outputs/plots/plotly_corr.html")