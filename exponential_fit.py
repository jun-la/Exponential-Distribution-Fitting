import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
import os
import pandas as pd
warnings.filterwarnings('ignore')


class ExponentialFitter:
    """
    A class to fit exponential distribution to data and perform statistical analysis.
    """
    
    def __init__(self, data=None):
        """
        Initialize the ExponentialFitter with optional data.
        
        Parameters:
        -----------
        data : array-like, optional
            Input data to fit exponential distribution
        """
        self.data = np.array(data) if data is not None else None
        self.lambda_hat = None
        self.fitted_dist = None
        self.fit_results = {}
        
    def generate_sample_data(self, lambda_param=2.0, sample_size=1000, random_state=None):
        """
        Generate sample data from exponential distribution.
        
        Parameters:
        -----------
        lambda_param : float
            Rate parameter of exponential distribution (λ > 0)
        sample_size : int
            Number of samples to generate
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        numpy.ndarray
            Generated sample data
        """
        if lambda_param <= 0:
            raise ValueError("Lambda parameter must be positive")
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")
            
        np.random.seed(random_state)
        self.data = np.random.exponential(scale=1/lambda_param, size=sample_size)
        return self.data
    
    def load_data_from_file(self, file_path, column=None, delimiter=None, skip_rows=0, 
                           header=None, encoding='utf-8'):
        """
        Load data from a text file (CSV, TSV, or space/tab separated).
        
        Parameters:
        -----------
        file_path : str
            Path to the text file containing data
        column : str or int, optional
            Column name or index to use. If None, uses the first column
        delimiter : str, optional
            Delimiter character. If None, attempts to auto-detect
        skip_rows : int, optional
            Number of rows to skip at the beginning of the file
        header : int, optional
            Row number to use as header (0-indexed). If None, no header is assumed
        encoding : str, optional
            File encoding (default: 'utf-8')
            
        Returns:
        --------
        numpy.ndarray
            Loaded data array
            
        Raises:
        -------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file is empty or contains invalid data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Try to read with pandas first (handles various formats automatically)
            if delimiter is None:
                # Try to auto-detect delimiter
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline().strip()
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif ',' in first_line:
                        delimiter = ','
                    elif ';' in first_line:
                        delimiter = ';'
                    else:
                        delimiter = None  # Let pandas auto-detect
            
            # Read the file
            if delimiter:
                df = pd.read_csv(file_path, delimiter=delimiter, skiprows=skip_rows, 
                               header=header, encoding=encoding)
            else:
                df = pd.read_csv(file_path, skiprows=skip_rows, header=header, 
                               encoding=encoding, sep=None, engine='python')
            

            
            # Select the appropriate column
            if column is not None:
                if isinstance(column, int):
                    if column >= len(df.columns):
                        raise ValueError(f"Column index {column} out of range. File has {len(df.columns)} columns.")
                    data = df.iloc[:, column].values
                else:
                    if column not in df.columns:
                        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
                    data = df[column].values
            else:
                # Use the first column
                data = df.iloc[:, 0].values
            
            # Convert to numeric and handle missing values
            data = pd.to_numeric(data, errors='coerce')
            data = data[~pd.isna(data)]  # Remove NaN values
            
            if len(data) == 0:
                raise ValueError("No valid numeric data found in the file")
            
            self.data = data
            return self.data
            
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or contains no valid data")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
    
    def load_data_from_simple_text(self, file_path, encoding='utf-8'):
        """
        Load data from a simple text file with one number per line.
        
        Parameters:
        -----------
        file_path : str
            Path to the text file containing data (one number per line)
        encoding : str, optional
            File encoding (default: 'utf-8')
            
        Returns:
        --------
        numpy.ndarray
            Loaded data array
            
        Raises:
        -------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file is empty or contains invalid data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            data = []
            with open(file_path, 'r', encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        try:
                            value = float(line)
                            data.append(value)
                        except ValueError:
                            print(f"Warning: Skipping invalid value on line {line_num}: '{line}'")
            
            if len(data) == 0:
                raise ValueError("No valid numeric data found in the file")
            
            self.data = np.array(data)
            return self.data
            
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")
    
    def fit_exponential(self, data=None):
        """
        Fit exponential distribution to data using maximum likelihood estimation.
        
        Parameters:
        -----------
        data : array-like, optional
            Data to fit. If None, uses self.data
            
        Returns:
        --------
        dict
            Dictionary containing fit results
        """
        if data is not None:
            self.data = np.array(data)
        
        if self.data is None:
            raise ValueError("No data provided for fitting")
        
        # Remove any non-positive values (exponential distribution is defined for x > 0)
        data_clean = self.data[self.data > 0]
        
        if len(data_clean) == 0:
            raise ValueError("No positive values in data for exponential fitting")
        
        # Maximum likelihood estimate for exponential distribution
        # λ̂ = 1 / x̄ where x̄ is the sample mean
        self.lambda_hat = 1 / np.mean(data_clean)
        
        # Create fitted distribution object
        self.fitted_dist = stats.expon(scale=1/self.lambda_hat)
        
        # Calculate additional statistics
        self.fit_results = {
            'lambda_hat': self.lambda_hat,
            'mean': np.mean(data_clean),
            'std': np.std(data_clean),
            'sample_size': len(data_clean),
            'original_size': len(self.data),
            'removed_negative': len(self.data) - len(data_clean)
        }
        
        return self.fit_results
    
    def goodness_of_fit_test(self, alpha=0.05):
        """
        Perform Kolmogorov-Smirnov goodness-of-fit test.
        
        Parameters:
        -----------
        alpha : float
            Significance level for the test
            
        Returns:
        --------
        dict
            Dictionary containing test results
        """
        if self.fitted_dist is None:
            raise ValueError("Must fit distribution before performing goodness-of-fit test")
        
        data_clean = self.data[self.data > 0]
        
        # Perform KS test
        ks_statistic, p_value = stats.kstest(data_clean, self.fitted_dist.cdf)
        
        test_results = {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'test_name': 'Kolmogorov-Smirnov'
        }
        
        return test_results
    
    def plot_fit(self, bins=50, figsize=(12, 8)):
        """
        Plot the fitted exponential distribution against the histogram of data.
        
        Parameters:
        -----------
        bins : int
            Number of histogram bins
        figsize : tuple
            Figure size (width, height)
        """
        if self.fitted_dist is None:
            raise ValueError("Must fit distribution before plotting")
        
        data_clean = self.data[self.data > 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram with fitted PDF
        ax1.hist(data_clean, bins=bins, density=True, alpha=0.7, color='skyblue', 
                label='Data Histogram')
        
        x_range = np.linspace(0, np.max(data_clean), 1000)
        ax1.plot(x_range, self.fitted_dist.pdf(x_range), 'r-', linewidth=2, 
                label=f'Fitted Exponential (λ={self.lambda_hat:.3f})')
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Exponential Distribution Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(data_clean, dist=self.fitted_dist, plot=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_summary_stats(self):
        """
        Get summary statistics of the fitted distribution.
        
        Returns:
        --------
        dict
            Dictionary containing summary statistics
        """
        if self.fitted_dist is None:
            raise ValueError("Must fit distribution before getting summary statistics")
        
        data_clean = self.data[self.data > 0]
        
        summary = {
            'sample_statistics': {
                'mean': np.mean(data_clean),
                'std': np.std(data_clean),
                'median': np.median(data_clean),
                'min': np.min(data_clean),
                'max': np.max(data_clean),
                'skewness': stats.skew(data_clean),
                'kurtosis': stats.kurtosis(data_clean)
            },
            'fitted_distribution': {
                'lambda_hat': self.lambda_hat,
                'theoretical_mean': 1 / self.lambda_hat,
                'theoretical_std': 1 / self.lambda_hat,
                'theoretical_median': np.log(2) / self.lambda_hat
            }
        }
        
        return summary


def fit_exponential_distribution(data, plot=True, bins=50):
    """
    Convenience function to fit exponential distribution to data.
    
    Parameters:
    -----------
    data : array-like
        Input data
    plot : bool
        Whether to create a plot
    bins : int
        Number of histogram bins for plotting
        
    Returns:
    --------
    ExponentialFitter
        Fitted ExponentialFitter object
    """
    fitter = ExponentialFitter(data)
    fitter.fit_exponential()
    
    if plot:
        fitter.plot_fit(bins=bins)
    
    return fitter


def fit_exponential_from_file(file_path, column=None, delimiter=None, skip_rows=0, 
                             header=None, encoding='utf-8', plot=True, bins=50):
    """
    Convenience function to load data from file and fit exponential distribution.
    
    Parameters:
    -----------
    file_path : str
        Path to the text file containing data
    column : str or int, optional
        Column name or index to use. If None, uses the first column
    delimiter : str, optional
        Delimiter character. If None, attempts to auto-detect
    skip_rows : int, optional
        Number of rows to skip at the beginning of the file
    header : int, optional
        Row number to use as header (0-indexed). If None, no header is assumed
    encoding : str, optional
        File encoding (default: 'utf-8')
    plot : bool
        Whether to create a plot
    bins : int
        Number of histogram bins for plotting
        
    Returns:
    --------
    ExponentialFitter
        Fitted ExponentialFitter object
    """
    fitter = ExponentialFitter()
    fitter.load_data_from_file(file_path, column, delimiter, skip_rows, header, encoding)
    fitter.fit_exponential()
    
    if plot:
        fitter.plot_fit(bins=bins)
    
    return fitter


def fit_exponential_from_simple_text(file_path, encoding='utf-8', plot=True, bins=50):
    """
    Convenience function to load data from simple text file and fit exponential distribution.
    
    Parameters:
    -----------
    file_path : str
        Path to the text file containing data (one number per line)
    encoding : str, optional
        File encoding (default: 'utf-8')
    plot : bool
        Whether to create a plot
    bins : int
        Number of histogram bins for plotting
        
    Returns:
    --------
    ExponentialFitter
        Fitted ExponentialFitter object
    """
    fitter = ExponentialFitter()
    fitter.load_data_from_simple_text(file_path, encoding)
    fitter.fit_exponential()
    
    if plot:
        fitter.plot_fit(bins=bins)
    
    return fitter


if __name__ == "__main__":
    # Example usage
    print("Exponential Distribution Fitting Example")
    print("=" * 50)
    
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
    
    # Goodness of fit test
    gof_results = fitter.goodness_of_fit_test()
    print(f"\nGoodness of Fit Test (Kolmogorov-Smirnov):")
    print(f"KS statistic: {gof_results['ks_statistic']:.4f}")
    print(f"p-value: {gof_results['p_value']:.4f}")
    print(f"Reject null hypothesis: {gof_results['reject_null']}")
    
    # Summary statistics
    summary = fitter.get_summary_stats()
    print(f"\nSummary Statistics:")
    print(f"Sample skewness: {summary['sample_statistics']['skewness']:.3f}")
    print(f"Sample kurtosis: {summary['sample_statistics']['kurtosis']:.3f}")
    print(f"Theoretical mean: {summary['fitted_distribution']['theoretical_mean']:.3f}")
    
    # Plot the results
    fitter.plot_fit()
