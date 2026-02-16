# visualize_codebleu.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# Configuration
# ------------------------------
input_csv = "./codebleu_results_analysis.csv"
output_folder = "visualizations"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# Load results CSV
# ------------------------------
df = pd.read_csv(input_csv)

# Compute similarity from distance
df['cb_similarity'] = 1.0 - df['cb_d_sim']

# ------------------------------
# Generate scatter plots per app type
# ------------------------------
for app_type, app_df in df.groupby("app_type"):

    # Save the data used for this app type
    csv_path = os.path.join(output_folder, f"{app_type}_scatter_data.csv")
    app_df[['cb_similarity', 'cb_d_perf']].to_csv(csv_path, index=False)
    print(f"Saved CSV for {app_type}: {csv_path}")

    # Scatter plot
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='cb_similarity', y='cb_d_perf', data=app_df, alpha=0.6)
    
    # Optional: add regression line
    sns.regplot(x='cb_similarity', y='cb_d_perf', data=app_df,
                scatter=False, color='red')

    plt.xlabel("CodeBLEU Similarity")
    plt.ylabel("Objective Distance")
    plt.title(f"{app_type}: CodeBLEU vs Objective Distance")
    plt.grid(True)

    # Save figure
    png_path = os.path.join(output_folder, f"{app_type}_scatter.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot for {app_type}: {png_path}")

print("All scatter plots and CSVs generated successfully!")
