import json
import matplotlib.pyplot as plt
import numpy as np

file_path = "../block_22215272.json"

# Initialize variables
data = []
current_block = ""

# Read the file and accumulate blocks of JSON data
with open(file_path, "r") as file:
    for line in file:
        if line.strip() == "}":  # End of a JSON block
            current_block += line.strip()
            try:
                # Try parsing the complete block
                data.append(json.loads(current_block))
            except json.JSONDecodeError as e:
                print(f"Error decoding block: {e}")
            current_block = ""  # Reset for the next block
        else:
            current_block += line.strip()  # Accumulate lines

# Check if data is loaded
print(f"Loaded {len(data)} blocks")

# Now, let's extract the timestamp (convert to seconds) and value (convert to Ether) to plot
timestamps = []
values = []

for entry in data:
    try:
        timestamp_ms = int(entry.get("timestamp_ms", 0))
        value_wei = int(entry.get("value", 0))
        value_eth = value_wei / 10**18  # Convert Wei to Ether
        timestamp_s = timestamp_ms / 1000  # Convert ms to seconds
        timestamps.append(timestamp_s)
        values.append(value_eth)
    except ValueError as e:
        print(f"Error extracting data: {e}")

# Sort the data by timestamp (ascending order)
sorted_data = sorted(zip(timestamps, values))

# Unzip the sorted data into separate lists
timestamps, values = zip(*sorted_data)

# Make timestamps relative to the first one (subtract the first timestamp)
relative_timestamps = np.array(timestamps) - timestamps[0]

# Create a bigger plot
plt.figure(figsize=(16, 12))  # Increase figure size

# Style improvements
plt.plot(relative_timestamps, values, label="Value (ETH)", color='darkblue', linewidth=3)

# Set x-ticks to show only whole numbers
plt.xticks(np.arange(0, int(np.max(relative_timestamps)) + 1, 1), fontsize=18, rotation=45)

# Add labels, title, grid, and customizations
plt.xlabel("Relative Time (s)", fontsize=24, weight='bold', labelpad=20)  # Increased font size and label padding
plt.ylabel("Value (ETH)", fontsize=24, weight='bold', labelpad=20)      # Increased font size and label padding
plt.title("Bids (ETH) vs. Time for Block 22215272", fontsize=26, weight='bold')

# Grid settings for better readability
plt.grid(True, which='both', linestyle='--', linewidth=1)

# Customize the tick labels for better visibility
plt.xticks(fontsize=22, rotation=45)
plt.yticks(fontsize=22)

# Add a legend with a bigger font
plt.legend(fontsize=22)

# Show the plot
plt.tight_layout()

# Save with a higher DPI (300 DPI for better resolution)
plt.savefig('timestamp_vs_value_plot.pdf', format='pdf', dpi=400)

