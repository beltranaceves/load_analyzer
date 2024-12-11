import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks, sawtooth, savgol_filter
from nicegui import ui


class Graph:
    def __init__(self):
        self.type = 'pure_sine'
        self.plot = None
        self.generate_sample_loads()
    
    def set_type(self, type):
        self.type = type
        self.generate_sample_loads()
        self.plot_graph.refresh()
        ui.notify(f"Graph type set to: {self.type}")

    def generate_sample_loads(self):
        """Generate different types of load patterns for testing"""
        t = np.linspace(0, 10, 500)
        if self.type == "pure_sine":
            pure_sine = 100 + 50 * np.sin(2 * np.pi * t)
            new_plot = pure_sine
        elif self.type == "noisy_sine":
            noisy_sine = 100 + 50 * np.sin(2 * np.pi * t) + np.random.normal(0, 10, 500)
            new_plot = noisy_sine
        elif self.type == "pure_noise":
            random_load = np.cumsum(np.random.normal(0, 10, 500))
            new_plot = random_load
        elif self.type == "loosely_sinusoidal":
            loosely_sinusoidal = (100 + 40 * np.sin(2 * np.pi * t) +
                                20 * np.sin(4 * np.pi * t) +
                                10 * np.random.normal(0, 1, 500))
            new_plot = loosely_sinusoidal
        elif self.type == "daily_pattern":
            daily_pattern = (100 + 30 * np.sin(2 * np.pi * t) +
                            20 * np.abs(np.sin(4 * np.pi * t)) +
                            np.random.normal(0, 5, 500))
            new_plot = daily_pattern
        elif self.type == "sawtooth_pattern":
            sawtooth_pattern = 100 + 50 * sawtooth(2 * np.pi * t) + np.random.normal(0, 5, 500)
            new_plot = sawtooth_pattern
        elif self.type == "step_pattern":
            steps = np.repeat([80, 120, 90, 140], 125)
            step_pattern = steps + np.random.normal(0, 5, 500)
            new_plot = step_pattern
        elif self.type == "exp_growth":
                exp_growth = (100 * np.exp(t/10) +
                            20 * np.sin(2 * np.pi * t) +
                            np.random.normal(0, 10, 500))
                new_plot = exp_growth
        self.plot = new_plot
        
    @ui.refreshable
    def plot_graph(self):
        if self.plot is not None:
            with ui.pyplot() as fig:
                plt.plot(self.plot)
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.title(f'{self.type} Graph')
                plt.close()
                
graph = Graph()
with ui.column(align_items= 'center') as left_column:
    with ui.row() as upper_row:
        with ui.column() as graph_column:
            with ui.row():
                select2 = ui.select(
                    {'pure_sine': 'Pure Sine Wave',
                    'noisy_sine': 'Noisy Sine Wave', 
                    'pure_noise': 'Pure Noise',
                    'loosely_sinusoidal': 'Loosely Sinusoidal',
                    'daily_pattern': 'Daily Pattern',
                    'sawtooth_pattern': 'Sawtooth Pattern',
                    'step_pattern': 'Step Pattern',
                    'exp_growth': 'Exponential Growth'},
                    label = 'Graph Type',
                    value = 'pure_sine',
                    with_input = True,
                    on_change = lambda e: graph.set_type(e.value)
                    ).bind_value(graph, 'type')
            with ui.row():
                graph.plot_graph()
        with ui.column() as graph_config_column:
            ui.label("This is a nicegui app")
    with ui.row() as lower_row:
        with ui.column():
            ui.label("This is a nicegui app")
        with ui.column():
            ui.label("This is a nicegui app")
        

if __name__ in {"__main__", "__mp_main__"}:
    ui.dark_mode().enable()
    ui.run()