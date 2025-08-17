#!/usr/bin/env python3
"""
Example usage of the exponential distribution fitting library.

This script demonstrates various features including:
- Data generation
- Distribution fitting
- Goodness of fit testing
- Statistical analysis
- Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from exponential_fit import ExponentialFitter, fit_exponential_distribution, fit_exponential_from_file, fit_exponential_from_simple_text

def example_1_basic_usage():
    """Example 1: Basic usage with generated data."""
    print("=" * 60)
    print("Example 1: Basic Usage with Generated Data")
    print("=" * 60)
    
    # Create fitter and generate sample data
    fitter = ExponentialFitter()
    data = fitter.generate_sample_data(lambda_param=1.5, sample_size=1000, random_state=42)
    
    print(f"Generated {len(data)} samples from exponential distribution with λ=1.5")
    print(f"Sample mean: {np.mean(data):.3f}")
    print(f"Sample std: {np.std(data):.3f}")
    
    # Fit exponential distribution
    results = fitter.fit_exponential()
    print(f"\nFitted λ̂: {results['lambda_hat']:.3f}")
    print(f"Fitted mean: {results['mean']:.3f}")
    print(f"Sample size used: {results['sample_size']}")
    
    # Goodness of fit test
    gof_results = fitter.goodness_of_fit_test()
    print(f"\nGoodness of Fit Test (Kolmogorov-Smirnov):")
    print(f"KS statistic: {gof_results['ks_statistic']:.4f}")
    print(f"p-value: {gof_results['p_value']:.4f}")
    print(f"Reject null hypothesis: {gof_results['reject_null']}")
    
    return fitter

def example_2_external_data():
    """Example 2: Fitting external data."""
    print("\n" + "=" * 60)
    print("Example 2: Fitting External Data")
    print("=" * 60)
    
    # Simulate external data (e.g., from file or experiment)
    np.random.seed(123)
    external_data = np.random.exponential(scale=1/0.8, size=500)
    
    print(f"External data: {len(external_data)} samples")
    print(f"Data mean: {np.mean(external_data):.3f}")
    print(f"Data std: {np.std(external_data):.3f}")
    
    # Fit using convenience function
    fitter = fit_exponential_distribution(external_data, plot=False)
    
    print(f"\nFitted λ̂: {fitter.lambda_hat:.3f}")
    print(f"Fitted mean: {fitter.fit_results['mean']:.3f}")
    
    # Check goodness of fit
    gof_results = fitter.goodness_of_fit_test()
    if gof_results['reject_null']:
        print("❌ Data does not follow exponential distribution")
    else:
        print("✅ Data is consistent with exponential distribution")
    
    return fitter

def example_3_comprehensive_analysis():
    """Example 3: Comprehensive statistical analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Comprehensive Statistical Analysis")
    print("=" * 60)
    
    # Generate data with different lambda
    fitter = ExponentialFitter()
    data = fitter.generate_sample_data(lambda_param=2.0, sample_size=2000, random_state=456)
    fitter.fit_exponential()
    
    # Get comprehensive summary statistics
    summary = fitter.get_summary_stats()
    
    print("Sample Statistics:")
    print("-" * 20)
    for key, value in summary['sample_statistics'].items():
        print(f"  {key:12}: {value:.3f}")
    
    print("\nFitted Distribution:")
    print("-" * 20)
    for key, value in summary['fitted_distribution'].items():
        print(f"  {key:18}: {value:.3f}")
    
    # Theoretical relationships
    lambda_hat = summary['fitted_distribution']['lambda_hat']
    theoretical_mean = 1 / lambda_hat
    theoretical_std = 1 / lambda_hat
    
    print(f"\nTheoretical Relationships:")
    print(f"  Mean = 1/λ = {theoretical_mean:.3f}")
    print(f"  Std = 1/λ = {theoretical_std:.3f}")
    print(f"  Mean ≈ Std (exponential property): {abs(theoretical_mean - theoretical_std) < 0.001}")
    
    return fitter

def example_4_edge_cases():
    """Example 4: Handling edge cases."""
    print("\n" + "=" * 60)
    print("Example 4: Edge Cases")
    print("=" * 60)
    
    # Test with mixed positive/negative data
    mixed_data = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 0.5, -1.5])
    print(f"Mixed data: {mixed_data}")
    
    fitter = ExponentialFitter()
    results = fitter.fit_exponential(mixed_data)
    
    print(f"Original data size: {results['original_size']}")
    print(f"Positive values used: {results['sample_size']}")
    print(f"Negative values removed: {results['removed_negative']}")
    print(f"Fitted λ̂: {results['lambda_hat']:.3f}")
    
    # Test with single value
    single_data = np.array([2.0])
    fitter_single = ExponentialFitter()
    results_single = fitter_single.fit_exponential(single_data)
    
    print(f"\nSingle value data: {single_data}")
    print(f"Fitted λ̂: {results_single['lambda_hat']:.3f}")
    print(f"Expected λ̂ = 1/2.0 = 0.5: {abs(results_single['lambda_hat'] - 0.5) < 0.001}")

def example_5_visualization():
    """Example 5: Visualization examples."""
    print("\n" + "=" * 60)
    print("Example 5: Visualization")
    print("=" * 60)
    
    # Generate data and create visualization
    fitter = ExponentialFitter()
    data = fitter.generate_sample_data(lambda_param=1.0, sample_size=1000, random_state=789)
    fitter.fit_exponential()
    
    print("Creating visualization...")
    print("(Close the plot window to continue)")
    
    # Create plot
    fig = fitter.plot_fit(bins=30, figsize=(14, 6))
    
    # Add some custom annotations
    ax1, ax2 = fig.axes
    
    # Add text annotation to histogram
    ax1.text(0.7, 0.8, f'λ̂ = {fitter.lambda_hat:.3f}', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.show()
    
    return fitter

def example_6_accuracy_comparison():
    """Example 6: Accuracy comparison with different sample sizes."""
    print("\n" + "=" * 60)
    print("Example 6: Accuracy Comparison")
    print("=" * 60)
    
    true_lambda = 0.5
    sample_sizes = [10, 50, 100, 500, 1000, 5000]
    
    print(f"True λ: {true_lambda}")
    print(f"{'Sample Size':<12} {'Estimated λ̂':<15} {'Error':<10} {'Relative Error %':<15}")
    print("-" * 60)
    
    for size in sample_sizes:
        fitter = ExponentialFitter()
        data = fitter.generate_sample_data(true_lambda, size, random_state=42)
        results = fitter.fit_exponential()
        
        estimated_lambda = results['lambda_hat']
        error = abs(estimated_lambda - true_lambda)
        relative_error = (error / true_lambda) * 100
        
        print(f"{size:<12} {estimated_lambda:<15.4f} {error:<10.4f} {relative_error:<15.2f}")

def example_7_file_reading():
    """Example 7: Reading data from files."""
    print("\n" + "=" * 60)
    print("Example 7: Reading Data from Files")
    print("=" * 60)
    
    # Check if sample files exist
    csv_file = "sample_data.csv"
    tsv_file = "sample_data.tsv"
    simple_file = "sample_data_simple.txt"
    
    print("Reading from CSV file...")
    if os.path.exists(csv_file):
        # Method 1: Using the convenience function
        fitter_csv = fit_exponential_from_file(csv_file, column='value', header=0, plot=False)
        print(f"CSV data: {len(fitter_csv.data)} samples")
        print(f"Fitted λ̂: {fitter_csv.lambda_hat:.3f}")
        
        # Method 2: Using the class method
        fitter2 = ExponentialFitter()
        data = fitter2.load_data_from_file(csv_file, column='value', header=0)
        results = fitter2.fit_exponential()
        print(f"Alternative method - Fitted λ̂: {results['lambda_hat']:.3f}")
    else:
        print(f"Sample CSV file '{csv_file}' not found")
    
    print("\nReading from TSV file...")
    if os.path.exists(tsv_file):
        fitter_tsv = fit_exponential_from_file(tsv_file, column='value', header=0, plot=False)
        print(f"TSV data: {len(fitter_tsv.data)} samples")
        print(f"Fitted λ̂: {fitter_tsv.lambda_hat:.3f}")
    else:
        print(f"Sample TSV file '{tsv_file}' not found")
    
    print("\nReading from simple text file...")
    if os.path.exists(simple_file):
        fitter_simple = fit_exponential_from_simple_text(simple_file, plot=False)
        print(f"Simple text data: {len(fitter_simple.data)} samples")
        print(f"Fitted λ̂: {fitter_simple.lambda_hat:.3f}")
        
        # Show summary statistics
        summary = fitter_simple.get_summary_stats()
        print(f"Sample mean: {summary['sample_statistics']['mean']:.3f}")
        print(f"Sample std: {summary['sample_statistics']['std']:.3f}")
    else:
        print(f"Sample simple text file '{simple_file}' not found")
    
    print("\nFile reading features demonstrated:")
    print("- CSV files with headers")
    print("- TSV files with headers") 
    print("- Simple text files (one number per line)")
    print("- Column selection by name or index")
    print("- Auto-delimiter detection")
    print("- Comment line handling (for simple text files)")

def main():
    """Run all examples."""
    print("Exponential Distribution Fitting - Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_external_data()
        example_3_comprehensive_analysis()
        example_4_edge_cases()
        example_5_visualization()
        example_6_accuracy_comparison()
        example_7_file_reading()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
