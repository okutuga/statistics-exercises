import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BikeKPIReport:
    def __init__(self, data):
        """
        Initializes the BikeKPIReport object.

        Args:
            data: A DataFrame containing the KPI data for the bikes.
        """
        self.data = data
        self.bike_colors = {
            "Ducati": "red",
            "KTM": "orange",
            "Yamaha": "blue",
            "Honda": "yellow",
            "Aprilia": "green"
        }

    def generate_html_report(self, output_file, plots):
        """
        Generates an interactive HTML report with multiple plots for the bikes.

        Args:
            output_file: The path to the output HTML file.
            plots: A list of dictionaries, each containing the x-axis data, y-axis data, title, and labels for a plot.
        """
        fig = make_subplots(rows=len(plots), cols=1, subplot_titles=[plot['title'] for plot in plots])

        for i, plot in enumerate(plots):
            x_data = plot['x_data']
            y_data = plot['y_data']
            x_label = plot['x_label']
            y_label = plot['y_label']

            # Add traces for each rider
            for rider in self.data['rider_number'].unique():
                rider_data = self.data[self.data['rider_number'] == rider]
                rider_label = f"{rider_data['rider_name'].iloc[0][0]}{rider_data['rider_surname'].iloc[0][0]}{rider}"
                bike_color = self.bike_colors.get(rider_data['bike_name'].iloc[0], 'black')
                fig.add_trace(go.Scatter(
                    x=rider_data[x_data],
                    y=rider_data[y_data],
                    mode='lines+markers',
                    name=f'{rider_label} - {plot["title"]}',
                    legendgroup=f'{rider_label}',
                    line=dict(color=bike_color)
                ), row=i + 1, col=1)

            # Update x and y axis labels
            fig.update_xaxes(title_text=x_label, row=i + 1, col=1)
            fig.update_yaxes(title_text=y_label, row=i + 1, col=1)

        # Update layout
        fig.update_layout(
            title='Bike KPIs Report',
            legend_title='Riders',
            template='plotly_white'
        )

        # Add interactive filter
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=["visible", "legendonly"],
                            label="Hide All",
                            method="restyle"
                        ),
                        dict(
                            args=["visible", True],
                            label="Show All",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ]
        )

        # Save the plot as an HTML file
        fig.write_html(output_file)
