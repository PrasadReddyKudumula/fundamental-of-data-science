import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data8-1.csv", names=['salary'])

# Histogram stats
hist_data, bin_edges = np.histogram(data, bins=32, range=[0, 148526])

# Center points and bin widths
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
bin_widths = bin_edges[1:] - bin_edges[:-1]

# PDF normalization
pdf_vals = hist_data / np.sum(hist_data)

# Cumulative distribution
cdf_vals = np.cumsum(pdf_vals)

# Plotting PDF
plt.figure(figsize=(8, 6))
plt.bar(bin_centers, pdf_vals, width=0.9 * bin_widths)
plt.xlabel('Claim value, £', fontsize=15)
plt.ylabel('Probability', fontsize=15)

# Mean calculation and plot
mean_value = np.sum(bin_centers * pdf_vals)
plt.plot([mean_value, mean_value], [0.0, max(pdf_vals)], c='red')
plt.text(x=35000, y=max(pdf_vals) - 0.005, s=f'Mean value: {int(mean_value)}', fontsize=12, color='red')

# Standard deviation
std_dev = np.sqrt(np.sum(((bin_centers - mean_value) ** 2) * pdf_vals))

# Standard deviation line
plt.axvline(std_dev, color='black', linestyle='--', linewidth=2)
plt.text(std_dev, max(pdf_vals), f'std: {std_dev:.2f}', color='black', horizontalalignment='right')

# Display plot
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Histogram with Mean')
plt.show()

print("Standard Deviation:", std_dev)
