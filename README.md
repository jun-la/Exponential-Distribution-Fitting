# Exponential Distribution Fitting

A Python library for fitting exponential distributions to data, generating sample data, and performing statistical analysis.

## Features

- **Data Generation**: Generate sample data from exponential distribution with specified parameters
- **Distribution Fitting**: Fit exponential distribution to data using maximum likelihood estimation
- **Goodness of Fit Testing**: Perform Kolmogorov-Smirnov goodness-of-fit tests
- **Statistical Analysis**: Calculate comprehensive summary statistics
- **Visualization**: Create histograms with fitted PDF and Q-Q plots
- **Comprehensive Testing**: Full unit test suite with edge cases

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from exponential_fit import ExponentialFitter

# Create fitter and generate sample data
fitter = ExponentialFitter()
data = fitter.generate_sample_data(lambda_param=1.5, sample_size=1000, random_state=42)

# Fit exponential distribution
results = fitter.fit_exponential()
print(f"Fitted λ̂: {results['lambda_hat']:.3f}")

# Perform goodness of fit test
gof_results = fitter.goodness_of_fit_test()
print(f"KS test p-value: {gof_results['p_value']:.4f}")

# Plot results
fitter.plot_fit()
```

### Reading Data from Files

```python
from exponential_fit import fit_exponential_from_file, fit_exponential_from_simple_text

# Read from CSV file with headers
fitter = fit_exponential_from_file("data.csv", column="values", header=0, plot=True)

# Read from simple text file (one number per line)
fitter = fit_exponential_from_simple_text("data.txt", plot=True)

# Read from CSV without headers (use column index)
fitter = fit_exponential_from_file("data.csv", column=1, plot=True)
```

## API Reference

### ExponentialFitter Class

#### Constructor
```python
ExponentialFitter(data=None)
```
- `data`: Optional array-like data to fit

#### Methods

##### `generate_sample_data(lambda_param=2.0, sample_size=1000, random_state=None)`
Generate sample data from exponential distribution.
- `lambda_param`: Rate parameter (λ > 0)
- `sample_size`: Number of samples
- `random_state`: Random seed for reproducibility

##### `fit_exponential(data=None)`
Fit exponential distribution using maximum likelihood estimation.
- `data`: Optional data to fit (uses self.data if None)
- Returns: Dictionary with fit results

##### `goodness_of_fit_test(alpha=0.05)`
Perform Kolmogorov-Smirnov goodness-of-fit test.
- `alpha`: Significance level
- Returns: Dictionary with test results

##### `plot_fit(bins=50, figsize=(12, 8))`
Create visualization of fitted distribution.
- `bins`: Number of histogram bins
- `figsize`: Figure size
- Returns: Matplotlib figure object

##### `get_summary_stats()`
Get comprehensive summary statistics.
- Returns: Dictionary with sample and theoretical statistics

##### `load_data_from_file(file_path, column=None, delimiter=None, skip_rows=0, header=None, encoding='utf-8')`
Load data from a text file (CSV, TSV, or space/tab separated).
- `file_path`: Path to the text file
- `column`: Column name or index to use (None = first column)
- `delimiter`: Delimiter character (None = auto-detect)
- `skip_rows`: Number of rows to skip
- `header`: Row number to use as header
- `encoding`: File encoding
- Returns: Loaded data array

##### `load_data_from_simple_text(file_path, encoding='utf-8')`
Load data from a simple text file with one number per line.
- `file_path`: Path to the text file
- `encoding`: File encoding
- Returns: Loaded data array

### Convenience Functions

```python
fit_exponential_distribution(data, plot=True, bins=50)
```
One-step function to fit exponential distribution to data.

```python
fit_exponential_from_file(file_path, column=None, delimiter=None, skip_rows=0, header=None, encoding='utf-8', plot=True, bins=50)
```
One-step function to load data from file and fit exponential distribution.

```python
fit_exponential_from_simple_text(file_path, encoding='utf-8', plot=True, bins=50)
```
One-step function to load data from simple text file and fit exponential distribution.

## Mathematical Background

### Exponential Distribution
The exponential distribution with rate parameter λ has:
- **PDF**: f(x) = λe^(-λx) for x ≥ 0
- **Mean**: E[X] = 1/λ
- **Variance**: Var[X] = 1/λ²
- **Standard Deviation**: σ = 1/λ

### Maximum Likelihood Estimation
For exponential distribution, the MLE of λ is:
- **λ̂ = 1/x̄** where x̄ is the sample mean

## Examples

### Example 1: Basic Usage
```python
import numpy as np
from exponential_fit import ExponentialFitter

# Generate data
fitter = ExponentialFitter()
data = fitter.generate_sample_data(lambda_param=0.5, sample_size=500)

# Fit distribution
results = fitter.fit_exponential()
print(f"True λ: 0.5, Estimated λ̂: {results['lambda_hat']:.3f}")
```

### Example 2: Fitting External Data
```python
# Load your data
your_data = [1.2, 0.8, 2.1, 1.5, 0.9, ...]

# Fit exponential distribution
fitter = ExponentialFitter(your_data)
results = fitter.fit_exponential()

# Check goodness of fit
gof_results = fitter.goodness_of_fit_test()
if gof_results['reject_null']:
    print("Data does not follow exponential distribution")
else:
    print("Data is consistent with exponential distribution")
```

### Example 3: Comprehensive Analysis
```python
# Generate and analyze data
fitter = ExponentialFitter()
data = fitter.generate_sample_data(lambda_param=2.0, sample_size=1000)
fitter.fit_exponential()

# Get summary statistics
summary = fitter.get_summary_stats()
print("Sample Statistics:")
for key, value in summary['sample_statistics'].items():
    print(f"  {key}: {value:.3f}")

print("\nFitted Distribution:")
for key, value in summary['fitted_distribution'].items():
    print(f"  {key}: {value:.3f}")

# Visualize
fitter.plot_fit()
```

### Example 4: Reading Data from Files
```python
# Read from CSV file with headers
fitter = fit_exponential_from_file("data.csv", column="values", header=0, plot=False)
print(f"Fitted λ̂: {fitter.lambda_hat:.3f}")

# Read from simple text file
fitter = fit_exponential_from_simple_text("data.txt", plot=False)
print(f"Fitted λ̂: {fitter.lambda_hat:.3f}")

# Read with custom parameters
fitter = ExponentialFitter()
data = fitter.load_data_from_file("data.csv", column=1, skip_rows=2, header=0)
results = fitter.fit_exponential()
```

## Running Tests

Run the complete test suite:
```bash
python test_exponential_fit.py
```

Run specific test classes:
```bash
python -m unittest test_exponential_fit.TestExponentialFitter
python -m unittest test_exponential_fit.TestStatisticalProperties
```

## Test Coverage

The test suite covers:
- ✅ Data generation with valid/invalid parameters
- ✅ Distribution fitting with various data types
- ✅ Edge cases (single values, mixed positive/negative data)
- ✅ Goodness of fit testing
- ✅ Statistical properties validation
- ✅ Visualization functionality
- ✅ Error handling and input validation
- ✅ Consistency and accuracy tests

## Dependencies

- **numpy**: Numerical computing and array operations
- **scipy**: Statistical functions and distributions
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation and file reading

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this library.
