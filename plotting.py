import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

FONTCOLOR = "rgba(0.4,0.4,0.4,1.0)"
GRIDCOLOR = "rgba(1.0,1.0,1.0,0.3)"
FILL_COLOR = "rgb(.7,.7,.7)"
COLORS = ["rgb(1, 138, 199)", "rgb(173, 214, 233)", "rgb(202, 110, 56)"]
COMPASS_COL_SCALE = [
    "rgb(1, 138, 199)",
    "rgb(105, 193, 246)",
    "rgb(255,255,255)",
    "rgb(202, 110, 56)",
    "rgb(202, 70, 56)",
][::-1]

FONTSIZE = 16
OPACITY = 0.6
BW = 0.1  # .1


def plotly_setting():
    pio.kaleido.scope.default_format = "svg"
    pio.templates.default = "plotly_white"


def plot_heatmap(
    corr,
    size=(600, 600),
    font_sz=14,
    blocksize=3,
    color_continuous_scale=COMPASS_COL_SCALE,
):  # px.colors.diverging.balance_r):
    mask = np.ones_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = False
    corr[mask.T] = np.nan

    fig = px.imshow(
        corr,
        text_auto="0.2f",
        color_continuous_scale=color_continuous_scale,
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        height=size[1],
        width=size[0],
        font=dict(size=font_sz),
    )
    fig.update_coloraxes(
        colorbar=dict(
            lenmode="fraction",
            len=0.5,
            thickness=12,
            x=1.01,
            xanchor="right",
            yanchor="top",
            y=0.845,
            orientation="h",
            # title="Correlation",
        ),
    )
    for i in range(4):
        p = i * blocksize - 0.5
        fig.add_shape(
            type="rect",
            x0=p,
            y0=p,
            x1=p,
            y1=blocksize * 3 - 0.5,
            line=dict(color="black", width=1.5),
            opacity=OPACITY,
        )
        fig.add_shape(
            type="rect",
            x0=-0.5,
            y0=p,
            x1=p,
            y1=p,
            line=dict(color="black", width=1.5),
            opacity=OPACITY,
        )

    fig["layout"]["margin"] = go.layout.Margin(l=0, r=0, b=0, t=0)
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    return fig


def plot_combined_distributions(sample1, sample2, xlabels, group_labels):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True)

    for i, x_label in enumerate(xlabels):
        fig.add_trace(
            go.Histogram(
                x=sample1[x_label],
                histnorm="probability density",
                name=group_labels[0],  # name used in legend and hover labels
                xbins=dict(start=-1, end=1, size=0.15),  # bins used for histogram
                marker_color=COLORS[0],
                opacity=OPACITY,
                legendgroup=group_labels[0] if i == 0 else "",
                showlegend=i == 0,
            ),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Histogram(
                x=sample2[x_label],
                histnorm="probability density",
                name=group_labels[1],
                xbins=dict(start=-1, end=1, size=0.15),  # bins used for histogram
                marker_color=COLORS[-1],
                opacity=OPACITY,
                legendgroup=group_labels[0] if i == 0 else "",
                showlegend=i == 0,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(
            range=[-1, 1],
            title_text=x_label,
            row=1,
            col=i + 1,
            tickangle=0,
            ticklabelstep=2,
        )

    fig.update_layout(
        # title_text="Sentiments through the session",
        yaxis=dict(title="Probability density"),
        font=dict(size=18),
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,
    )  # gap between bars of the same location coordinates)

    return fig


def plot_timeseries(samples1, samples2, xlabels, group_labels):
    fig = make_subplots(
        rows=3, cols=1, shared_yaxes=True, shared_xaxes=True, x_title="Time Index"
    )
    for idx, (xlabel, sample1, sample2) in enumerate(
        zip(
            xlabels,
            samples1,
            samples2,
        )
    ):
        sl = True if idx == 0 else False
        plt = go.Scatter(
            x=sample1[0],
            y=sample1[1],
            fill="tozeroy",
            name=group_labels[0],
            showlegend=sl,
            mode="lines",
            line=dict(color=COLORS[0]),
            opacity=OPACITY,
        )
        plt1 = go.Scatter(
            x=sample2[0],
            y=sample2[1],
            fill="tozeroy",
            name=group_labels[1],
            showlegend=sl,
            mode="lines",
            line=dict(color=COLORS[-1]),
            opacity=OPACITY,
        )

        fig.add_trace(plt, row=idx + 1, col=1)
        fig.add_trace(plt1, row=idx + 1, col=1)

    for idx, label in enumerate(xlabels):
        fig["layout"][f"yaxis{idx + 1}"]["title"] = f"{label.capitalize()}"

    fig.update_layout(
        width=700,
        height=200 * len(xlabels),
        font=dict(size=FONTSIZE, color=FONTCOLOR),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font_size=FONTSIZE - 2, yanchor="top", y=1.1, xanchor="right", x=0.99
        ),
    )
    fig["layout"]["annotations"][0]["font"]["size"] = FONTSIZE + 4
    fig.update_yaxes(
        range=[-0.5, 0.5], showgrid=False, showline=False, zerolinecolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(
        showgrid=False,
        showline=False,
        zerolinecolor="rgba(0.5,0.5,0.5,0.3)",
        ticklabelposition="outside right",
    )

    return fig
