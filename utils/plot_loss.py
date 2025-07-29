import re
import matplotlib.pyplot as plt

def plot_losses_from_log(log_file: str):
    # Initialise each type of loss
    train_loss = []  
    train_loss_mean = {}
    val_loss = {}
    val_loss_mean = {}
    test_loss = None
    max_window_index = 0  # Track the highest training window number seen

    # Patterns
    train_loss_pattern = re.compile(r"TRAINING LOSS FOR PATCH \d+, RUN \d+, EPOCH (\d+), TRAINING WINDOW (\d+): ([0-9.]+)")
    train_loss_mean_pattern = re.compile(r"MEAN train MAE loss \(across all windows of current epoch\), RUN \d+, EPOCH (\d+): ([0-9.]+)")
    val_loss_pattern = re.compile(r"VALIDATION LOSS FOR PATCH \d+, RUN \d+, EPOCH (\d+): ([0-9.]+)")
    val_loss_mean_pattern = re.compile(r"MEAN val MAE loss \(across 1st epoch to current epoch\), RUN \d+, EPOCH (\d+): ([0-9.]+)")
    test_loss_pattern = re.compile(r"FINAL test MAE loss \(last epoch\), RUN \d+, EPOCH \d+: ([0-9.]+)")

    # Parse log file
    raw_train_loss = []
    with open(log_file, 'r') as f:
        for line in f:
            # Train loss per window 
            if match := train_loss_pattern.search(line):
                epoch = int(match.group(1))
                window = int(match.group(2))
                loss = float(match.group(3))

                if window > max_window_index:
                    max_window_index = window

                raw_train_loss.append((epoch, window, loss))
                continue

            # Train loss per epoch (mean across all windows)
            if match := train_loss_mean_pattern.search(line):
                epoch = int(match.group(1))
                loss = float(match.group(2))
                train_loss_mean[epoch] = loss
                continue

            # Validation loss per epoch
            if match := val_loss_pattern.search(line):
                epoch = int(match.group(1))
                loss = float(match.group(2))
                val_loss[epoch] = loss
                continue

            # Validation loss per epoch (mean across epochs so far)
            if match := val_loss_mean_pattern.search(line):
                epoch = int(match.group(1))
                loss = float(match.group(2))
                val_loss_mean[epoch] = loss
                continue

            # Test loss
            if match := test_loss_pattern.search(line):
                if test_loss is None:
                    test_loss = float(match.group(1))

    # Compute fractional epoch values
    train_loss = [((epoch - 1) + window / max_window_index, loss) for epoch, window, loss in raw_train_loss]

    # Sort and unpack
    train_loss.sort()
    x_train, y_train = zip(*train_loss) if train_loss else ([], [])
    all_epochs = sorted(set(train_loss_mean) | set(val_loss) | set(val_loss_mean))
    x_epochs = list(all_epochs)
    y_train_mean = [train_loss_mean.get(e, None) for e in x_epochs]
    y_val = [val_loss.get(e, None) for e in x_epochs]
    y_val_mean = [val_loss_mean.get(e, None) for e in x_epochs]

    # Construct labels using final values for each curve
    final_epoch = x_epochs[-1] if x_epochs else None
    final_train_mean = train_loss_mean.get(final_epoch)
    final_val = val_loss.get(final_epoch)
    final_val_mean = val_loss_mean.get(final_epoch)
    train_mean_label = "Training loss, mean across windows of an epoch" + f" (final = {final_train_mean:.4f})"
    val_label = "Validation loss" + f" (final = {final_val:.4f})"
    val_mean_label = "Validation loss, mean across epochs" + f" (final = {final_val_mean:.4f})"

    # Plotting
    plt.figure(figsize=(12, 7))
    
    plt.plot(x_train, y_train, label="Training loss", color='gray', linestyle='-', alpha=0.6)
    plt.plot(x_epochs, y_train_mean, label=train_mean_label, marker='o')
    plt.plot(x_epochs, y_val, label=val_label, marker='x')
    plt.plot(x_epochs, y_val_mean, label=val_mean_label, marker='^')
    plt.axhline(y=test_loss, color='red', linestyle='--', label=f'Test loss = {test_loss:.4f}')

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    output_path = log_file.replace('.log', '_loss_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
