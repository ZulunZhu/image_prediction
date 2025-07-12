import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "/Users/cheryltay/Documents/project_dpm5/NoNeighbor_NodeFeatOnly/patch_0004/logs/logger_20250709_214125.log"

# Lists to hold the extracted losses
train_losses = []
val_losses = []
test_loss = None  # Initialize test_loss

# Regex patterns
train_pattern = re.compile(r"train mean Training MAE loss, ([0-9.]+)")
val_pattern = re.compile(r"validate mean val MAE loss, ([0-9.]+)")
test_pattern = re.compile(r"test test_end MAE loss, ([0-9.]+)")

# Read and extract losses
with open(log_file, 'r') as file:
    for line in file:
        train_match = train_pattern.search(line)
        val_match = val_pattern.search(line)
        test_match = test_pattern.search(line)

        if train_match:
            train_losses.append(float(train_match.group(1)))
        if val_match:
            val_losses.append(float(val_match.group(1)))
        if test_match and test_loss is None:  # Capture only the first test loss
            test_loss = float(test_match.group(1))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training MAE Loss', marker='o')
plt.plot(val_losses, label='Validation MAE Loss', marker='x')

# Add test loss as a horizontal line if found
if test_loss is not None:
    plt.axhline(y=test_loss, color='red', linestyle='--', label=f'Test MAE Loss = {test_loss:.4f}')

plt.ylim(0, 0.45)
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("MAE Loss", fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(log_file.replace('.log', '_loss_plot.png'))
