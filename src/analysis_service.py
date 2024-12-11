import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.fft import fft
from scipy.signal import correlate, find_peaks, sawtooth, savgol_filter

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
    

    def _smooth_data(self, window_length=21, polyorder=3):
        """
        Smooth the load data using Savitzky-Golay filter
        
        Parameters:
        -----------
        window_length : int, optional
            Length of the filter window (must be odd)
        polyorder : int, optional
            Order of the polynomial used to fit the samples
        """
        if window_length % 2 == 0:
            window_length += 1  # Ensure window length is odd
        return savgol_filter(self.load_data, window_length, polyorder)
    def _find_chunks(self):
        """Find chunks between low points in the signal"""
        # Find local minima
        low_points, _ = find_peaks(-self.load_data)
        
        if len(low_points) < 2:
            return [self.load_data]  # Return single chunk if not enough low points
            
        # Split data into chunks
        chunks = []
        for i in range(len(low_points)-1):
            chunk = self.load_data[low_points[i]:low_points[i+1]]
            if len(chunk) > 10:  # Only keep chunks with sufficient points
                chunks.append(chunk)
                
        return chunks
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
        """
        # Compute autocorrelation
        autocorr = correlate(self.load_data, self.load_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize autocorrelation
        autocorr = autocorr / autocorr[0]
        
        # Find peaks with stricter parameters
        peaks, properties = find_peaks(autocorr, 
                                     height=0.2,      # Minimum height threshold
                                     prominence=0.1)   # Minimum prominence
        
        if len(peaks) > 1:
            peak_distances = np.diff(peaks)
            peak_heights = properties['peak_heights'][1:]  # Exclude first peak
            
            # Calculate regularity using both distance and height variations
            distance_variation = np.std(peak_distances) / np.mean(peak_distances)
            height_variation = np.std(peak_heights) / np.mean(peak_heights)
            
            # Combined variation coefficient (weighted average)
            variation_coefficient = (distance_variation + height_variation) / 2
            
            peak_regularity = {
                'mean_period': np.mean(peak_distances),
                'period_variation_coefficient': variation_coefficient
            }
        else:
            peak_regularity = {
                'mean_period': None,
                'period_variation_coefficient': 1.0  # Set to maximum irregularity
            }
        
        return peak_regularity
    
    def statistical_analysis(self):
        """Compute statistical properties including chunk-based kurtosis"""
        chunks = self._find_chunks()
        chunk_kurtosis = []
        
        # Calculate kurtosis for each chunk
        for chunk in chunks:
            k = stats.kurtosis(chunk)
            if not np.isnan(k):  # Ignore invalid results
                chunk_kurtosis.append(k)
        
        # Use median kurtosis if chunks exist, else use global kurtosis
        final_kurtosis = np.median(chunk_kurtosis) if chunk_kurtosis else stats.kurtosis(self.load_data)
        
        return {
            'mean': np.mean(self.load_data),
            'standard_deviation': np.std(self.load_data),
            'skewness': stats.skew(self.load_data),
            'kurtosis': final_kurtosis,
            'chunk_kurtosis': chunk_kurtosis
        }    
    def is_approximately_sinusoidal(self, tolerance=0.2):
        """
        Determine if the load pattern is approximately sinusoidal
        """
        # Apply smoothing to data before analysis
        original_data = self.load_data
        self.load_data = self._smooth_data()
    
        # Perform analyses
        freq_analysis = self.frequency_analysis()
        period_analysis = self.periodicity_analysis()
        stat_analysis = self.statistical_analysis()
    
        # Restore original data
        self.load_data = original_data
    
        # Rest of the function remains the same...
        criteria_met = [
            freq_analysis['frequency_dominance_ratio'] > (1 - tolerance),
            (period_analysis['period_variation_coefficient'] is not None and 
         period_analysis['period_variation_coefficient'] < tolerance),
            abs(stat_analysis['skewness']) < tolerance,
            abs(stat_analysis['kurtosis'] - 1.5) < tolerance
        ]
    
        analysis_results = {
            'frequency_analysis': freq_analysis,
            'periodicity_analysis': period_analysis,
            'statistical_analysis': stat_analysis,
            'criteria_met': criteria_met,
            'sinusoidal_score': np.mean(criteria_met),
            'is_sinusoidal': np.mean(criteria_met) > (1 - tolerance)
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
                     f'({"Sinusoidal" if analysis_results["is_sinusoidal"] else "Non-Sinusoidal"})')
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
    t = np.linspace(0, 10, 500)
    
    # Existing patterns
    pure_sine = 100 + 50 * np.sin(2 * np.pi * t)
    noisy_sine = 100 + 50 * np.sin(2 * np.pi * t) + np.random.normal(0, 10, 500)
    random_load = np.cumsum(np.random.normal(0, 10, 500))
    
    # New patterns
    # 1. Loosely sinusoidal (combination of multiple frequencies)
    loosely_sinusoidal = (100 + 40 * np.sin(2 * np.pi * t) + 
                         20 * np.sin(4 * np.pi * t) + 
                         10 * np.random.normal(0, 1, 500))
    
    # 2. Daily pattern with peaks (like web traffic)
    daily_pattern = (100 + 30 * np.sin(2 * np.pi * t) + 
                    20 * np.abs(np.sin(4 * np.pi * t)) + 
                    np.random.normal(0, 5, 500))
    
    # 3. Sawtooth pattern (like batch processing)
    sawtooth_pattern = 100 + 50 * sawtooth(2 * np.pi * t) + np.random.normal(0, 5, 500)
    
    # 4. Step function with noise (like service deployment)
    steps = np.repeat([80, 120, 90, 140], 125)
    step_pattern = steps + np.random.normal(0, 5, 500)
    
    # 5. Exponential growth with periodic fluctuation
    exp_growth = (100 * np.exp(t/10) + 
                 20 * np.sin(2 * np.pi * t) + 
                 np.random.normal(0, 10, 500))
    
    return {
        'pure_sine': pure_sine,
        'noisy_sine': noisy_sine,
        'random_load': random_load,
        'loosely_sinusoidal': loosely_sinusoidal,
        'daily_pattern': daily_pattern,
        'sawtooth_pattern': sawtooth_pattern,
        'step_pattern': step_pattern,
        'exp_growth': exp_growth
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