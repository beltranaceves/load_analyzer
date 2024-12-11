from nicegui import ui
import numpy as np
with ui.row() as upper_row:
    with ui.column() as graph_column:
        with ui.row():
            with ui.dropdown_button('Select Graph', auto_close=True) as graph_dropdown:
                ui.item('Pure Sine Wave', on_click=lambda: ui.notify('You clicked item 1'))
                ui.item('Noisy Sine Wave', on_click=lambda: ui.notify('You clicked item 2'))
                ui.item('Pure Noise Wave', on_click=lambda: ui.notify('You clicked item 2'))
        with ui.row():
            with ui.matplotlib().figure as fig:
                x = np.linspace(0.0, 5.0)
                y = np.cos(2 * np.pi * x) * np.exp(-x)
                ax = fig.gca()
                ax.plot(x, y, '-')
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