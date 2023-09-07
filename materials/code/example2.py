import numpy as np

def categorize_by_quantile(arr):
    # Define quantile thresholds (20th, 40th, 60th, 80th percentiles)
    quantile_thresholds = np.percentile(arr, [20, 40, 60, 80])

    # Categorize elements based on quantile thresholds
    quantile_categories = np.digitize(arr, quantile_thresholds)

    # Map numeric categories to "Q1", "Q2", "Q3", "Q4", "Q5"
    quantile_labels = [f"Q{category + 1}" for category in quantile_categories]

    return quantile_labels

# Sample array
arr = np.array([-0.4658, 0.0617, -0.8360, -0.4911, 0.6145, -0.9189, -1, 1, -0.8343, -0.0972])

# Categorize elements by quantile
quantile_labels = categorize_by_quantile(arr)

print("Quantile categories:", quantile_labels)
