from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def delay_histogram(schedule: pd.DataFrame):
    if schedule.empty:
        fig = go.Figure()
        fig.update_layout(title="Delay Histogram")
        fig.add_annotation(text="No delay data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig
    return px.histogram(
        schedule,
        x="delay",
        nbins=20,
        title="Delay Histogram",
    )
