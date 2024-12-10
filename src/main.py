import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.fft import fft
from scipy.signal import correlate, find_peaks

class ServiceLoadAnalyzer:
    def __init__(self, load_data, time_intervals=None):
        """
        Initialize the analyzer with service load time series data
        
        Parameters:
        -----------
        load_data : array-like
            Time series of service load measurements
        time_intervals : array-like, optional
            Corresponding time points for each measurement
        """
        self.load_data = np.array(load_data)
        self.time_intervals = time_intervals if time_intervals is not None else np.arange(len(load_data))
    
    def frequency_analysis(self):
        """
        Perform Fourier Transform to analyze frequency components
        
        Returns:
        --------
        dict: Detailed frequency analysis results
        """
        # Perform Fast Fourier Transform
        fft_result = np.abs(fft(self.load_data))
        
        # Normalize and compute frequencies
        n = len(self.load_data)
        frequencies = np.fft.fftfreq(n)
        
        # Sort frequencies by magnitude (excluding zero frequency)
        sorted_indices = np.argsort(fft_result[1:n//2])[::-1] + 1
        top_frequencies = sorted_indices[:3]
        top_magnitudes = fft_result[top_frequencies]
        
        return {
            'dominant_frequency': frequencies[top_frequencies[0]],
            'frequency_magnitudes': top_magnitudes,
            'frequency_dominance_ratio': top_magnitudes[0] / np.sum(top_magnitudes[1:])
        }
    
    def periodicity_analysis(self):
        """
        Analyze periodicity using autocorrelation and peak detection
        
        Returns:
        --------
        dict: Detailed periodicity analysis results
        """
        # Compute autocorrelation
        autocorr = correlate(self.load_data, self.load_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr, height=0)
        
        # Compute peak distances
        if len(peaks) > 1:
            peak_distances = np.diff(peaks)
            peak_regularity = {
                'mean_period': np.mean(peak_distances),
                'period_variation_coefficient': np.std(peak_distances) / np.mean(peak_distances)
            }
        else:
            peak_regularity = {
                'mean_period': None,
                'period_variation_coefficient': None
            }
        
        return peak_regularity
    
    def statistical_analysis(self):
        """
        Compute statistical properties of the load data
        
        Returns:
        --------
        dict: Statistical properties
        """
        return {
            'mean': np.mean(self.load_data),
            'standard_deviation': np.std(self.load_data),
            'skewness': stats.skew(self.load_data),
            'kurtosis': stats.kurtosis(self.load_data)
        }
    
    def is_approximately_sinusoidal(self, tolerance=0.2):
        """
        Determine if the load pattern is approximately sinusoidal
        
        Parameters:
        -----------
        tolerance : float, optional
            Acceptable deviation from ideal sinusoidal characteristics
        
        Returns:
        --------
        bool: Whether the load pattern is approximately sinusoidal
        dict: Detailed analysis results
        """
        # Frequency analysis
        freq_analysis = self.frequency_analysis()
        
        # Periodicity analysis
        period_analysis = self.periodicity_analysis()
        
        # Statistical analysis
        stat_analysis = self.statistical_analysis()
        
        # Sinusoidal criteria checks
        criteria_met = [
            # Frequency dominance
            freq_analysis['frequency_dominance_ratio'] > (1 - tolerance),
            
            # Periodicity regularity
            (period_analysis['period_variation_coefficient'] is not None and 
             period_analysis['period_variation_coefficient'] < tolerance),
            
            # Skewness close to zero
            abs(stat_analysis['skewness']) < tolerance,
            
            # Kurtosis close to 1.5 (for sine wave)
            abs(stat_analysis['kurtosis'] - 1.5) < tolerance
        ]
        
        # Compile detailed results
        analysis_results = {
            'frequency_analysis': freq_analysis,
            'periodicity_analysis': period_analysis,
            'statistical_analysis': stat_analysis,
            'criteria_met': criteria_met,
            'sinusoidal_score': np.mean(criteria_met)
        }
        
        return np.mean(criteria_met) > (1 - tolerance), analysis_results
    
    def visualize(self, analysis_results=None):
        """
        Visualize the load data and analysis results
        
        Parameters:
        -----------
        analysis_results : dict, optional
            Results from is_approximately_sinusoidal method
        """
        plt.figure(figsize=(12, 10))
        
        # Original data plot
        plt.subplot(3, 1, 1)
        plt.plot(self.time_intervals, self.load_data)
        plt.title('Service Load Time Series')
        plt.xlabel('Time')
        plt.ylabel('Load')
        
        # FFT plot
        plt.subplot(3, 1, 2)
        fft_result = np.abs(fft(self.load_data))
        frequencies = np.fft.fftfreq(len(self.load_data))
        plt.plot(frequencies[:len(frequencies)//2], fft_result[:len(frequencies)//2])
        plt.title('Frequency Spectrum (FFT)')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        
        # Sinusoidal characteristics and score
        if analysis_results:
            plt.subplot(3, 1, 3)
            criteria_labels = ['Frequency\nDominance', 'Periodicity', 'Skewness', 'Kurtosis']
            bars = plt.bar(criteria_labels, analysis_results['criteria_met'])
            plt.title(f'Sinusoidal Characteristics\nScore: {analysis_results["sinusoidal_score"]:.2f} ' + 
                     f'({"Sinusoidal" if analysis_results["sinusoidal_score"] > 0.5 else "Non-Sinusoidal"})')
            plt.ylabel('Criterion Met')
            plt.ylim(0, 1.2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("analysis_results.png")
        plt.show()
        plt.close()
# Example usage demonstration
def generate_sample_loads():
    """Generate different types of load patterns for testing"""
    # Pure sinusoidal load
    t = np.linspace(0, 10, 500)
    pure_sine = 100 + 50 * np.sin(2 * np.pi * t)
    
    # Noisy sinusoidal load
    noisy_sine = 100 + 50 * np.sin(2 * np.pi * t) + np.random.normal(0, 10, 500)
    
    # Non-sinusoidal load
    random_load = np.cumsum(np.random.normal(0, 10, 500))
    
    return {
        'pure_sine': pure_sine,
        'noisy_sine': noisy_sine,
        'random_load': random_load
    }

# Demonstration
if __name__ == "__main__":
    loads = generate_sample_loads()
    
    for name, load in loads.items():
        print(f"\nAnalyzing {name} pattern:")
        analyzer = ServiceLoadAnalyzer(load)
        is_sinusoidal, results = analyzer.is_approximately_sinusoidal(0.5)
        
        print(f"Approximately Sinusoidal: {is_sinusoidal}")
        print(f"Sinusoidal Score: {results['sinusoidal_score']:.2f}")
        
        analyzer.visualize(results)
