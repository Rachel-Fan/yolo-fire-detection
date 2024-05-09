import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV files
df_yolov5 = pd.read_csv('yolov5/runs/train/exp3/results.csv')
df_yolov8n = pd.read_csv('runs/detect/train/results.csv')

# Strip whitespace from headers
df_yolov5.columns = df_yolov5.columns.str.strip()
df_yolov8n.columns = df_yolov8n.columns.str.strip()

# Map YOLOv8n columns to YOLOv5 column names for comparison
column_mapping = {
    'metrics/precision(B)': 'metrics/precision',
    'metrics/recall(B)': 'metrics/recall',
    'metrics/mAP50(B)': 'metrics/mAP_0.5',
    'metrics/mAP50-95(B)': 'metrics/mAP_0.5:0.95',
    'train/cls_loss': 'train/cls_loss',
    'train/box_loss': 'train/box_loss',
    'val/box_loss': 'val/box_loss',
    'val/cls_loss': 'val/cls_loss'
}

# Rename YOLOv8n columns to match YOLOv5 columns
df_yolov8n.rename(columns=column_mapping, inplace=True)

# List of dataframes and their labels
dfs = [df_yolov5, df_yolov8n]
labels = ['YOLOv5', 'YOLOv8n']

# Define the metrics and their titles
metrics = [
    ('metrics/precision', 'Precision'),
    ('metrics/recall', 'Recall'),
    ('metrics/mAP_0.5', 'mAP_0.5'),
    ('metrics/mAP_0.5:0.95', 'mAP_0.5:0.95'),
    ('train/box_loss', 'Training Box Loss'),
    ('train/obj_loss', 'Training Object Loss'),
    ('train/cls_loss', 'Training Class Loss'),
    ('val/box_loss', 'Validation Box Loss'),
    ('val/obj_loss', 'Validation Object Loss'),
    ('val/cls_loss', 'Validation Class Loss')
]

# Create a figure and a grid of subplots
fig, axs = plt.subplots(4, 3, figsize=(20, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot each metric
for i, (metric, title) in enumerate(metrics):
    ax = axs[i//3, i%3]
    for j, df in enumerate(dfs):
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], label=labels[j])
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend()

# Remove empty subplots
for i in range(len(metrics), 12):
    fig.delaxes(axs[i//3, i%3])

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), 'yolov5_vs_yolov8n_performance_plots.png')
plt.savefig(output_path)

# Display the plot
plt.show()

# Extract final epoch data
final_data = []
for df in dfs:
    df.columns = df.columns.str.strip()  # Strip any whitespace from column names
    final_data.append(df.iloc[-1])

# Create summary table
summary_table = pd.DataFrame(final_data, columns=[
    'epoch', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
    'train/box_loss', 'train/obj_loss', 'train/cls_loss',
    'val/box_loss', 'val/obj_loss', 'val/cls_loss'
], index=labels)

# Print summary table
print(summary_table)