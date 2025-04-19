import os
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Set paths to the coherence folders
work_dir = os.getcwd()
cor_folder = "/aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL"
output_file = os.path.join(work_dir, "plots/overall_plot.png")  

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

# Extract acquisition dates from the folder structure using Regex
cor_pattern = re.compile(r'cor_(\d{8})_(\d{8})')

# Initialise set to store unique acquisition dates
unique_dates = set()

# Extract unique acquisition dates from the folder structure
for folder in os.listdir(cor_folder):
    match = cor_pattern.match(folder)
    if match:
        date1, date2 = match.groups()
        unique_dates.update([date1, date2])

# Convert unique dates to a sorted list of datetime objects
acquisition_dates = sorted(unique_dates)                                    # Sorting unique dates in ascending order
acquisition_dates = pd.to_datetime(acquisition_dates, format="%Y%m%d")      # Converting to datetime objects
acquisition_dates_for_plotting = pd.Series(acquisition_dates)               # For plotting purposes only

# Detect and insert missing acquisitions (expected interval is 12 days)
expected_gap = pd.Timedelta(days=12)
filled_dates = [acquisition_dates[0]]

for i in range(1, len(acquisition_dates)):
    previous_date = filled_dates[-1]
    current_date = acquisition_dates[i]

    # If gap > 12 days, insert missing acquisitions
    while (current_date - previous_date) > expected_gap:
        previous_date += expected_gap           # Add 12 days to the last known date
        filled_dates.append(previous_date)      # Insert missing date

    filled_dates.append(current_date)

# Convert back to series for plotting
plotting_dates = pd.Series(filled_dates)

# Extract coherence pairs from the folder structure for plotting
coherence_pairs = []
for folder in os.listdir(cor_folder):
    match = cor_pattern.match(folder)
    if match:
        date1, date2 = match.groups()
        coherence_pairs.append((date1, date2))

# Calculate the number of days since the first acquisition for each acquisition date
start_date = plotting_dates.iloc[0]
days_since_start_plotting = (plotting_dates - start_date).dt.days
days_since_start_acquisitions = (acquisition_dates_for_plotting - start_date).dt.days

# Prepare the plot and make it squarish
fig, ax = plt.subplots(figsize=(9, 9))  

# Plot Sentinel-1 SLC acquisitions as black squares on the diagonal
ax.scatter(days_since_start_acquisitions, days_since_start_acquisitions, marker='s', s=100, color='black', label='SLC Acquisition')

# Plot coherence pairs as blue dots above the diagonal
for date1, date2 in coherence_pairs:
    x1 = (pd.to_datetime(date1, format="%Y%m%d") - start_date).days
    x2 = (pd.to_datetime(date2, format="%Y%m%d") - start_date).days
    
    # Skip plotting dots for the automatically detected missing acquisitions
    if pd.to_datetime(date1, format="%Y%m%d") not in acquisition_dates.values or \
       pd.to_datetime(date2, format="%Y%m%d") not in acquisition_dates.values:
        continue  # Leave space empty
    
    # x2 is plotted before x1 to ensure that the dots lie above the diagonal. If plotting below, swap the positions of x2 and x1 in the scatterplot.
    ax.scatter(x2, x1, marker='o', color='blue', alpha=0.7)  

# Set axis ticks
ax.set_xticks(days_since_start_plotting)
ax.set_yticks(days_since_start_plotting)

# Append date and timestep under each unique date, maintaining YYYYMMDD format. Assign timestep numbers every 12 days for regularity, even for skipped acquisitions.
tick_labels = [f"{date.strftime('%Y%m%d')}" for i, date in enumerate(plotting_dates)]
ax.set_xticklabels(tick_labels, rotation=90)
ax.set_yticklabels(tick_labels)

# Set axis labels and title
ax.set_xlabel("SLC Acquisition Dates")
ax.set_ylabel("SLC Acquisition Dates")
ax.set_title("Sentinel-1 SLCs and Coherence Pairs")

# Legend entries for SLC Acquisition (squares) and Coherence Pairs (dots)
slc_legend = Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label='SLC Acquisition')
coherence_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Coherence Pair')

# Add both legend entries to the same legend
ax.legend(handles=[slc_legend, coherence_legend], loc='upper right')

# Include gridlines for readability
ax.grid(True, linestyle="-", alpha=0.25)

# Invert Y-axis (earlier acquisitions at the top, later acquisitions at the bottom)
ax.invert_yaxis()

plt.tight_layout()

# Save the figure
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')  

# Show the figure (optional)
plt.show()

print(f"Figure saved as {output_file}")
