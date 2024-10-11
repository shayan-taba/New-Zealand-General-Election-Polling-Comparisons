import numpy as np
import pandas as pd
import scipy.stats as stats

# Load expected results from CSV
pollster_df = pd.read_csv('pollsters.csv', dtype={'Pollster': str, 'Nat': float, 'Lab': float, 'Grn': float, 'Act': float, 'NZF': float, 'TPM': float})

# Load actual results from CSV
actual_df = pd.read_csv('actual_results.csv', dtype={'Nat': float, 'Lab': float, 'Grn': float, 'Act': float, 'NZF': float, 'TPM': float})

# Assuming there is only one row in actual results, extract the actual values
actual_values = actual_df.iloc[0].values.astype(float)  # Ensure the data is float

# Prepare a list to hold the results
results = []

# Calculate SSD, R^2, and Chi-Squared for each pollster
for index, row in pollster_df.iterrows():
    pollster_name = row['Pollster']
    expected_values = row[1:].values.astype(float)  # Ensure the expected values are float
    
    # Calculate SSD
    ssd = np.sum((actual_values - expected_values) ** 2)
    
    # Calculate R^2
    mean_actual = np.mean(actual_values)
    ss_total = np.sum((actual_values - mean_actual) ** 2)
    ss_residual = np.sum((actual_values - expected_values) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Normalize expected values for Chi-Squared Test
    observed_sum = np.sum(actual_values)
    expected_sum = np.sum(expected_values)
    
    if observed_sum != expected_sum:
        scale_factor = observed_sum / expected_sum
        expected_values = expected_values * scale_factor
    
    # Calculate Chi-Squared
    chi2, p_value = stats.chisquare(f_obs=actual_values, f_exp=expected_values)
    
    # Append results to the list
    results.append({
        'Pollster': pollster_name,
        'SSD': ssd,
        'R^2': r_squared,
        'Chi-Squared': chi2,
        'P-Value': p_value
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('polling_analysis_results.csv', index=False)

print("Results have been saved to polling_analysis_results.csv")
