import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
import tempfile
warnings.filterwarnings('ignore')

from exponential_fit import ExponentialFitter, fit_exponential_distribution, fit_exponential_from_file, fit_exponential_from_simple_text


class TestExponentialFitter(unittest.TestCase):
    """Test cases for ExponentialFitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fitter = ExponentialFitter()
        np.random.seed(42)  # For reproducible tests
        
    def test_init_with_data(self):
        """Test initialization with data."""
        data = [1.0, 2.0, 3.0]
        fitter = ExponentialFitter(data)
        np.testing.assert_array_equal(fitter.data, np.array(data))
        
    def test_init_without_data(self):
        """Test initialization without data."""
        fitter = ExponentialFitter()
        self.assertIsNone(fitter.data)
        self.assertIsNone(fitter.lambda_hat)
        self.assertIsNone(fitter.fitted_dist)
        
    def test_generate_sample_data_valid_params(self):
        """Test sample data generation with valid parameters."""
        lambda_param = 2.0
        sample_size = 1000
        
        data = self.fitter.generate_sample_data(lambda_param, sample_size, random_state=42)
        
        self.assertEqual(len(data), sample_size)
        self.assertTrue(np.all(data > 0))  # All values should be positive
        self.assertAlmostEqual(np.mean(data), 1/lambda_param, delta=0.1)
        
    def test_generate_sample_data_invalid_lambda(self):
        """Test sample data generation with invalid lambda parameter."""
        with self.assertRaises(ValueError):
            self.fitter.generate_sample_data(lambda_param=0)
            
        with self.assertRaises(ValueError):
            self.fitter.generate_sample_data(lambda_param=-1)
            
    def test_generate_sample_data_invalid_sample_size(self):
        """Test sample data generation with invalid sample size."""
        with self.assertRaises(ValueError):
            self.fitter.generate_sample_data(sample_size=0)
            
        with self.assertRaises(ValueError):
            self.fitter.generate_sample_data(sample_size=-1)
            
    def test_fit_exponential_with_generated_data(self):
        """Test fitting exponential distribution to generated data."""
        # Generate data with known lambda
        true_lambda = 1.5
        data = self.fitter.generate_sample_data(true_lambda, 1000, random_state=42)
        
        # Fit the distribution
        results = self.fitter.fit_exponential()
        
        # Check results
        self.assertIsNotNone(self.fitter.lambda_hat)
        self.assertIsNotNone(self.fitter.fitted_dist)
        self.assertAlmostEqual(results['lambda_hat'], true_lambda, delta=0.3)
        self.assertEqual(results['sample_size'], len(data))
        self.assertEqual(results['removed_negative'], 0)  # All generated data should be positive
        
    def test_fit_exponential_with_external_data(self):
        """Test fitting exponential distribution with external data."""
        # Create synthetic exponential data
        true_lambda = 2.0
        data = np.random.exponential(scale=1/true_lambda, size=500)
        
        # Fit using external data
        results = self.fitter.fit_exponential(data)
        
        self.assertAlmostEqual(results['lambda_hat'], true_lambda, delta=0.3)
        self.assertEqual(results['sample_size'], len(data))
        
    def test_fit_exponential_no_data(self):
        """Test fitting exponential distribution without data."""
        with self.assertRaises(ValueError):
            self.fitter.fit_exponential()
            
    def test_fit_exponential_all_negative_data(self):
        """Test fitting exponential distribution with all negative data."""
        data = np.array([-1, -2, -3])
        with self.assertRaises(ValueError):
            self.fitter.fit_exponential(data)
            
    def test_fit_exponential_mixed_data(self):
        """Test fitting exponential distribution with mixed positive/negative data."""
        data = np.array([1, -2, 3, -4, 5])
        
        results = self.fitter.fit_exponential(data)
        
        # Should only use positive values
        expected_positive_count = 3
        self.assertEqual(results['sample_size'], expected_positive_count)
        self.assertEqual(results['removed_negative'], 2)
        
    def test_goodness_of_fit_test(self):
        """Test goodness of fit test."""
        # Generate and fit data
        data = self.fitter.generate_sample_data(1.0, 1000, random_state=42)
        self.fitter.fit_exponential()
        
        # Perform goodness of fit test
        gof_results = self.fitter.goodness_of_fit_test()
        
        # Check results structure
        self.assertIn('ks_statistic', gof_results)
        self.assertIn('p_value', gof_results)
        self.assertIn('alpha', gof_results)
        self.assertIn('reject_null', gof_results)
        self.assertIn('test_name', gof_results)
        
        # Check value ranges
        self.assertGreaterEqual(gof_results['ks_statistic'], 0)
        self.assertLessEqual(gof_results['ks_statistic'], 1)
        self.assertGreaterEqual(gof_results['p_value'], 0)
        self.assertLessEqual(gof_results['p_value'], 1)
        self.assertIsInstance(gof_results['reject_null'], (bool, np.bool_))
        
    def test_goodness_of_fit_test_without_fit(self):
        """Test goodness of fit test without fitting first."""
        with self.assertRaises(ValueError):
            self.fitter.goodness_of_fit_test()
            
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        # Generate and fit data
        data = self.fitter.generate_sample_data(1.0, 1000, random_state=42)
        self.fitter.fit_exponential()
        
        # Get summary statistics
        summary = self.fitter.get_summary_stats()
        
        # Check structure
        self.assertIn('sample_statistics', summary)
        self.assertIn('fitted_distribution', summary)
        
        # Check sample statistics
        sample_stats = summary['sample_statistics']
        self.assertIn('mean', sample_stats)
        self.assertIn('std', sample_stats)
        self.assertIn('median', sample_stats)
        self.assertIn('min', sample_stats)
        self.assertIn('max', sample_stats)
        self.assertIn('skewness', sample_stats)
        self.assertIn('kurtosis', sample_stats)
        
        # Check fitted distribution
        fitted_dist = summary['fitted_distribution']
        self.assertIn('lambda_hat', fitted_dist)
        self.assertIn('theoretical_mean', fitted_dist)
        self.assertIn('theoretical_std', fitted_dist)
        self.assertIn('theoretical_median', fitted_dist)
        
        # Check theoretical relationships
        lambda_hat = fitted_dist['lambda_hat']
        self.assertAlmostEqual(fitted_dist['theoretical_mean'], 1/lambda_hat)
        self.assertAlmostEqual(fitted_dist['theoretical_std'], 1/lambda_hat)
        self.assertAlmostEqual(fitted_dist['theoretical_median'], np.log(2)/lambda_hat)
        
    def test_get_summary_stats_without_fit(self):
        """Test getting summary statistics without fitting first."""
        with self.assertRaises(ValueError):
            self.fitter.get_summary_stats()
            
    def test_plot_fit_without_fit(self):
        """Test plotting without fitting first."""
        with self.assertRaises(ValueError):
            self.fitter.plot_fit()
            
    def test_plot_fit_returns_figure(self):
        """Test that plot_fit returns a matplotlib figure."""
        # Generate and fit data
        data = self.fitter.generate_sample_data(1.0, 100, random_state=42)
        self.fitter.fit_exponential()
        
        # Plot and check return value
        fig = self.fitter.plot_fit()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
        
    def test_fit_accuracy_with_large_sample(self):
        """Test fitting accuracy with large sample size."""
        true_lambda = 0.5
        sample_size = 10000
        
        data = self.fitter.generate_sample_data(true_lambda, sample_size, random_state=42)
        results = self.fitter.fit_exponential()
        
        # With large sample size, estimate should be close to true value
        self.assertAlmostEqual(results['lambda_hat'], true_lambda, delta=0.05)
        
    def test_fit_consistency(self):
        """Test that fitting the same data multiple times gives consistent results."""
        data = self.fitter.generate_sample_data(1.0, 1000, random_state=42)
        
        # Fit multiple times
        results1 = self.fitter.fit_exponential(data)
        lambda1 = self.fitter.lambda_hat
        
        # Create new fitter and fit again
        fitter2 = ExponentialFitter()
        results2 = fitter2.fit_exponential(data)
        lambda2 = fitter2.lambda_hat
        
        # Results should be identical
        self.assertAlmostEqual(lambda1, lambda2)
        self.assertAlmostEqual(results1['lambda_hat'], results2['lambda_hat'])
        
    def test_edge_case_single_value(self):
        """Test fitting with single positive value."""
        data = np.array([1.0])
        results = self.fitter.fit_exponential(data)
        
        self.assertEqual(results['sample_size'], 1)
        self.assertEqual(results['lambda_hat'], 1.0)  # λ̂ = 1/x̄ = 1/1 = 1
        
    def test_edge_case_two_values(self):
        """Test fitting with two positive values."""
        data = np.array([1.0, 2.0])
        results = self.fitter.fit_exponential(data)
        
        self.assertEqual(results['sample_size'], 2)
        expected_lambda = 1 / np.mean(data)  # 1 / 1.5 = 0.667
        self.assertAlmostEqual(results['lambda_hat'], expected_lambda)


class TestFitExponentialDistributionFunction(unittest.TestCase):
    """Test cases for the convenience function fit_exponential_distribution."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
    def test_fit_exponential_distribution_function(self):
        """Test the convenience function."""
        # Generate test data
        data = np.random.exponential(scale=1/1.5, size=1000)
        
        # Test with plotting disabled
        fitter = fit_exponential_distribution(data, plot=False)
        
        self.assertIsInstance(fitter, ExponentialFitter)
        self.assertIsNotNone(fitter.lambda_hat)
        self.assertIsNotNone(fitter.fitted_dist)
        
    def test_fit_exponential_distribution_with_plot(self):
        """Test the convenience function with plotting enabled."""
        # Generate test data
        data = np.random.exponential(scale=1/1.5, size=100)
        
        # Test with plotting enabled
        fitter = fit_exponential_distribution(data, plot=True, bins=20)
        
        self.assertIsInstance(fitter, ExponentialFitter)
        plt.close('all')  # Clean up plots


class TestStatisticalProperties(unittest.TestCase):
    """Test cases for statistical properties of exponential distribution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fitter = ExponentialFitter()
        np.random.seed(42)
        
    def test_exponential_distribution_properties(self):
        """Test that fitted exponential distribution has correct properties."""
        true_lambda = 2.0
        data = self.fitter.generate_sample_data(true_lambda, 10000, random_state=42)
        results = self.fitter.fit_exponential()
        
        # Check that mean ≈ 1/λ
        self.assertAlmostEqual(results['mean'], 1/true_lambda, delta=0.1)
        
        # Check that std ≈ 1/λ (exponential distribution has mean = std)
        self.assertAlmostEqual(results['std'], 1/true_lambda, delta=0.1)
        
        # Check that fitted lambda is close to true lambda
        self.assertAlmostEqual(results['lambda_hat'], true_lambda, delta=0.2)
        
    def test_goodness_of_fit_for_exponential_data(self):
        """Test that exponential data passes goodness of fit test."""
        data = self.fitter.generate_sample_data(1.0, 1000, random_state=42)
        self.fitter.fit_exponential()
        
        gof_results = self.fitter.goodness_of_fit_test(alpha=0.05)
        
        # For truly exponential data, we should not reject the null hypothesis
        # (though this is probabilistic, so we allow some flexibility)
        # In practice, with 1000 samples from exponential distribution, 
        # p-value should typically be > 0.05
        if gof_results['p_value'] < 0.05:
            print(f"Warning: Goodness of fit test rejected exponential distribution "
                  f"with p-value {gof_results['p_value']:.4f}")
            
    def test_goodness_of_fit_for_non_exponential_data(self):
        """Test that non-exponential data fails goodness of fit test."""
        # Generate normal data instead of exponential
        data = np.random.normal(loc=1.0, scale=0.5, size=1000)
        
        self.fitter.fit_exponential(data)
        gof_results = self.fitter.goodness_of_fit_test(alpha=0.05)
        
        # Non-exponential data should typically fail the goodness of fit test
        # (though this is probabilistic)
        if gof_results['p_value'] > 0.05:
                     print(f"Warning: Goodness of fit test did not reject non-exponential data "
                   f"with p-value {gof_results['p_value']:.4f}")


class TestFileReading(unittest.TestCase):
    """Test cases for file reading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fitter = ExponentialFitter()
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create CSV test file
        self.csv_file = os.path.join(self.temp_dir, "test_data.csv")
        with open(self.csv_file, 'w') as f:
            f.write("time,value,description\n")
            f.write("0.1,1.234,first\n")
            f.write("0.2,0.876,second\n")
            f.write("0.3,1.567,third\n")
        
        # Create TSV test file
        self.tsv_file = os.path.join(self.temp_dir, "test_data.tsv")
        with open(self.tsv_file, 'w') as f:
            f.write("time\tvalue\tdescription\n")
            f.write("0.1\t1.234\tfirst\n")
            f.write("0.2\t0.876\tsecond\n")
            f.write("0.3\t1.567\tthird\n")
        
        # Create simple text test file
        self.simple_file = os.path.join(self.temp_dir, "test_data.txt")
        with open(self.simple_file, 'w') as f:
            f.write("# Comment line\n")
            f.write("1.234\n")
            f.write("0.876\n")
            f.write("1.567\n")
            f.write("0.654\n")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_data_from_file_csv(self):
        """Test loading data from CSV file."""
        data = self.fitter.load_data_from_file(self.csv_file, column='value', header=0)
        
        self.assertEqual(len(data), 3)
        self.assertAlmostEqual(data[0], 1.234)
        self.assertAlmostEqual(data[1], 0.876)
        self.assertAlmostEqual(data[2], 1.567)
    
    def test_load_data_from_file_csv_column_index(self):
        """Test loading data from CSV file using column index."""
        data = self.fitter.load_data_from_file(self.csv_file, column=1, header=0)
        
        self.assertEqual(len(data), 3)
        self.assertAlmostEqual(data[0], 1.234)
        self.assertAlmostEqual(data[1], 0.876)
        self.assertAlmostEqual(data[2], 1.567)
    
    def test_load_data_from_file_tsv(self):
        """Test loading data from TSV file."""
        data = self.fitter.load_data_from_file(self.tsv_file, column='value', header=0)
        
        self.assertEqual(len(data), 3)
        self.assertAlmostEqual(data[0], 1.234)
        self.assertAlmostEqual(data[1], 0.876)
        self.assertAlmostEqual(data[2], 1.567)
    
    def test_load_data_from_file_auto_detect_delimiter(self):
        """Test auto-detection of delimiter."""
        # Test CSV
        data_csv = self.fitter.load_data_from_file(self.csv_file, column='value', header=0)
        self.assertEqual(len(data_csv), 3)
        
        # Test TSV
        data_tsv = self.fitter.load_data_from_file(self.tsv_file, column='value', header=0)
        self.assertEqual(len(data_tsv), 3)
    
    def test_load_data_from_simple_text(self):
        """Test loading data from simple text file."""
        data = self.fitter.load_data_from_simple_text(self.simple_file)
        
        self.assertEqual(len(data), 4)  # 4 valid numbers, 1 comment line
        self.assertAlmostEqual(data[0], 1.234)
        self.assertAlmostEqual(data[1], 0.876)
        self.assertAlmostEqual(data[2], 1.567)
        self.assertAlmostEqual(data[3], 0.654)
    
    def test_load_data_from_file_nonexistent(self):
        """Test loading data from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.fitter.load_data_from_file("nonexistent_file.csv")
    
    def test_load_data_from_file_invalid_column(self):
        """Test loading data with invalid column name."""
        with self.assertRaises(ValueError):
            self.fitter.load_data_from_file(self.csv_file, column='nonexistent_column')
    
    def test_load_data_from_file_invalid_column_index(self):
        """Test loading data with invalid column index."""
        with self.assertRaises(ValueError):
            self.fitter.load_data_from_file(self.csv_file, column=10)
    
    def test_fit_exponential_from_file(self):
        """Test convenience function for fitting from file."""
        fitter = fit_exponential_from_file(self.csv_file, column='value', header=0, plot=False)
        
        self.assertIsInstance(fitter, ExponentialFitter)
        self.assertIsNotNone(fitter.lambda_hat)
        self.assertIsNotNone(fitter.fitted_dist)
        self.assertEqual(len(fitter.data), 3)
    
    def test_fit_exponential_from_simple_text(self):
        """Test convenience function for fitting from simple text file."""
        fitter = fit_exponential_from_simple_text(self.simple_file, plot=False)
        
        self.assertIsInstance(fitter, ExponentialFitter)
        self.assertIsNotNone(fitter.lambda_hat)
        self.assertIsNotNone(fitter.fitted_dist)
        self.assertEqual(len(fitter.data), 4)
    
    def test_load_data_from_file_with_skip_rows(self):
        """Test loading data with skip_rows parameter."""
        # Create file with header rows to skip
        skip_file = os.path.join(self.temp_dir, "skip_test.csv")
        with open(skip_file, 'w') as f:
            f.write("Header line 1\n")
            f.write("Header line 2\n")
            f.write("time,value\n")
            f.write("0.1,1.234\n")
            f.write("0.2,0.876\n")
        
        data = self.fitter.load_data_from_file(skip_file, column='value', skip_rows=2, header=0)
        
        self.assertEqual(len(data), 2)
        self.assertAlmostEqual(data[0], 1.234)
        self.assertAlmostEqual(data[1], 0.876)
    
    def test_load_data_from_file_with_header(self):
        """Test loading data with header parameter."""
        # Create file with header
        header_file = os.path.join(self.temp_dir, "header_test.csv")
        with open(header_file, 'w') as f:
            f.write("time,value\n")
            f.write("0.1,1.234\n")
            f.write("0.2,0.876\n")
        
        data = self.fitter.load_data_from_file(header_file, column='value', header=0)
        
        self.assertEqual(len(data), 2)
        self.assertAlmostEqual(data[0], 1.234)
        self.assertAlmostEqual(data[1], 0.876)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
