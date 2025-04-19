import os
import numpy as np
import pandas as pd
import re
import shutil
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# ~~~~~~~~~~ SETTINGS (For Testing Only) ~~~~~~~~~~~~

data_dir = "/aria/zulun_sharing/TEST_KZ_TGNN_sharing_20250214/merged/cors_ALL"
temporal_window_index = 1   # Start at the first window iteration
temporal_window_width = 10  # Number of adjacent time steps one window covers.
temporal_window_stride = 1  # How far the window moves per step (set to `10` for disjoint windows)


def extract_all_dates(data_dir):
    """
    Extracts unique timestamps from folder names in the data directory.
    """
    # List all subdirectories
    folder_list = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    # Extract unique dates from folder names
    all_dates = set()
    for folder in folder_list:
        match = re.match(r"cor_(\d{8})_(\d{8})", folder)
        if match:
            all_dates.update(match.groups())  # Add both dates from the folder name

    # Sort dates in ascending order and create an indexed list
    timesteps = list(enumerate(sorted(all_dates)))
    return timesteps


def select_temporal_window_dates(timesteps, temporal_window_index, temporal_window_width, temporal_window_stride = 1):
    """
    Computes the subset of timesteps for a given sliding window position.
    """
    temporal_window_start_feature_index = (temporal_window_index - 1) * temporal_window_stride              # -1 adjustment needed for 0-based indexing
    temporal_window_end_feature_index = temporal_window_start_feature_index + temporal_window_width + 1     

    # Ensure the temporal window does not exceed available data
    if temporal_window_start_feature_index >= len(timesteps):
        print("WARNING: Window start index exceeds available data. Returning empty list.")
        return []
    
    # Extract the selected dates from the timesteps
    selected_dates = [date for _, date in timesteps[temporal_window_start_feature_index : temporal_window_end_feature_index]]
    return selected_dates


def filter_folders(data_dir, selected_dates, sorting_order = "second_date"):
    """
    Filters folders that contain only date pairs within the selected time window.
    """
    folder_list = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    selected_dates_set = set(selected_dates)  # Convert to set for faster lookup

    # Initialise empty list to store filtered folders
    filtered_folders = []

    # Iterate through each folder and check if both dates are within the selected range
    for folder in folder_list:
        match = re.match(r"cor_(\d{8})_(\d{8})", folder)  # Extract both date groups
        if match:
            date1, date2 = match.groups()  # Get the two dates from the folder name

            # Both dates must be within the selected range
            if date1 in selected_dates_set and date2 in selected_dates_set:
                filtered_folders.append((folder, date1, date2))

    # Sorting logic for the dates.
    # If train-val-test split is done according to the NODE FEATURES, then the SECOND DATE should be used for sorting.
    # If train-val-test split is done according to the EDGE FEATURES, then the FIRST DATE should be used for sorting.
    if sorting_order == "first_date":                      
        filtered_folders.sort(key=lambda x: (x[1], x[2]))  # Sort by first date in ascending order, then second date in ascending order (default).
    elif sorting_order == "second_date":
        filtered_folders.sort(key=lambda x: (x[2], x[1]))  # Sort by second date in ascending, then first date in ascending order.
    return [folder[0] for folder in filtered_folders]


# Identify the skipped dates in the selected window
def identify_skipped_timesteps(data_dir, expected_interval = 12):
    """
    Identify the skipped timesteps in the selected window.
    """
    # Extract all dates from the data directory
    timesteps = extract_all_dates(data_dir)

    # Convert to datetime objects
    datetime_steps = [datetime.strptime(date, '%Y%m%d') for date in timesteps]
    
    # Acquire the list of intervals between the dates e.g., [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    intervals = [(datetime_steps[i+1] - datetime_steps[i]).days for i in range(len(datetime_steps)-1)]

    # Count the number of missed intervals
    missed_intervals = [i for i, days in enumerate(intervals) if days != expected_interval]
    num_missed_intervals = len(missed_intervals)

    return missed_intervals, num_missed_intervals



# ~~~~~~~~~~ EXECUTING FUNCTIONS (For Testing) ~~~~~~~~~~~~

if __name__ == "__main__":
    timesteps = extract_all_dates(data_dir)
    selected_dates = select_temporal_window_dates(timesteps, temporal_window_index, temporal_window_width, temporal_window_stride)
    filtered_folders = filter_folders(data_dir, selected_dates, sorting_order = "second_date")

    # Print results
    print(f"\nTimesteps used for the selected window:\n", selected_dates)   # Display filtered dates for the selected window
    print(f"\nFiltered folders for the selected Window ({len(filtered_folders)} folders):")
    for index, folder in enumerate(filtered_folders):
        print(f"{index + 1}: {folder}")
        

