import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV files
df1 = pd.read_csv('runs/detect/train/results.csv')
df2 = pd.read_csv('runs/detect/train2/results.csv')
df3 = pd.read_csv('runs/detect/train3/results.csv')

# Strip whitespace from headers
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df3.columns = df3.columns.str.strip()

# Strip whitespace from headers
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df3.columns = df3.columns.str.strip()

# List of dataframes
dfs = [df1, df2, df3]

# Model names
model_names = ['YOLOv8n 20 epoch', 'YOLOv8n 100 epoch', 'YOLOv8m 20 epoch']

# Define the metrics and their titles
metrics = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP50'),
    ('metrics/mAP50-95(B)', 'mAP50-95'),
    ('train/box_loss', 'Training Box Loss'),
    ('train/cls_loss', 'Training Class Loss'),
    ('train/dfl_loss', 'Training DFL Loss'),
    ('val/box_loss', 'Validation Box Loss'),
    ('val/cls_loss', 'Validation Class Loss'),
    ('val/dfl_loss', 'Validation DFL Loss')
]

# Create a figure and a grid of subplots
fig, axs = plt.subplots(4, 3, figsize=(20, 12))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# Plot each metric
for i, (metric, title) in enumerate(metrics):
    ax = axs[i//3, i%3]
    for j, df in enumerate(dfs):
        ax.plot(df['epoch'], df[metric], label=model_names[j])
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.legend()

# Remove empty subplots
for i in range(len(metrics), 12):
    fig.delaxes(axs[i//3, i%3])

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), 'yolov8_performance_plots.png')
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
    'epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
])

# Print summary table
print(summary_table)