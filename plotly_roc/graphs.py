from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc

from .metrics import cm_table

HOVERTOOL_FONT_FACE = {"font": {"family": "Courier New, monospace"}}


def roc_curve(
    metrics_df: pd.DataFrame,
    fig: Optional[go.Figure] = None,
    line_name: str = None,
    line_color: str = "steelblue",
    cm_labels: List[str] = None,
    fig_size: Tuple[int, int] = (650, 500),
) -> go.Figure:
    """Create interactive plotly ROC curve

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The metrics dataframe from `metrics.metrics_df`
    fig : go.Figure, optional
        Figure to add line to. Provide to plot multiple curves on a single graph
    line_name : str, optional
        Name of the line, displayed in the legend
    line_color : str, optional
        color of line, by default "steelblue"
    cm_labels : List[str], optional
        Description of the binary labels, ie ["CAT", "DOG"], by default None
    fig_size : Tuple[int, int], optional
        Figure size, by default (650, 500)

    Returns
    -------
    go.Figure
        The plotly figure handle
    """
    cm_kwargs = dict()
    if cm_labels is not None:
        cm_kwargs["neg_label"] = cm_labels[0]
        cm_kwargs["pos_label"] = cm_labels[1]

    tooltips = metrics_df.apply(lambda row: cm_table(row, **cm_kwargs,), axis=1,)

    if fig is None:
        fig = go.Figure()

    if line_name:
        line_name += ";"
    else:
        line_name = ""

    auc_value = auc(metrics_df["FPR"], metrics_df["REC"])
    fig.add_trace(
        go.Scatter(
            x=metrics_df["FPR"].values,
            y=metrics_df["REC"].values,
            hovertext=tooltips,
            hoverlabel=HOVERTOOL_FONT_FACE,
            hoverinfo="text",
            marker=dict(color=line_color),
            name=f"{line_name} AUC: {auc_value:0.4f}",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            marker=dict(color="black"),
            line=dict(dash="dash"),
            showlegend=False,
        )
    )

    fig.update_layout(
        autosize=False,
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=fig_size[0],
        height=fig_size[1],
    )

    return fig


def precision_recall_curve(
    metrics_df: pd.DataFrame,
    fig: Optional[go.Figure] = None,
    line_name: str = None,
    line_color: str = "green",
    cm_labels: List[str] = None,
    fig_size: (int, int) = (700, 500),
) -> go.Figure:
    """Create interactive plotly precision recall curve

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The metrics dataframe from `metrics.metrics_df`
    fig : go.Figure, optional
        Figure to add line to. Provide to plot multiple curves on a single graph
    line_name : str, optional
        Name of the line, displayed in the legend
    line_color : str, optional
        color of line, by default "green"
    cm_labels : List[str], optional
        Description of the binary labels, ie ["CAT", "DOG"] to include in the confusion
        matrix tooltip
    fig_size : Tuple[int, int], optional
        Figure size, by default (650, 500)

    Returns
    -------
    go.Figure
        The plotly figure handle
    """

    cm_kwargs = dict()
    if cm_labels is not None:
        cm_kwargs["neg_label"] = cm_labels[0]
        cm_kwargs["pos_label"] = cm_labels[1]

    tooltips = metrics_df.apply(lambda row: cm_table(row, **cm_kwargs,), axis=1,)

    cm_kwargs = dict()
    if cm_labels is not None:
        cm_kwargs["neg_label"] = cm_labels[0]
        cm_kwargs["pos_label"] = cm_labels[1]

    if fig is None:
        fig = go.Figure()

    if line_name:
        line_name += ";"
    else:
        line_name = ""

    auc_value = auc(metrics_df["FPR"], metrics_df["REC"])
    fig.add_trace(
        go.Scatter(
            x=metrics_df["REC"].values,
            y=metrics_df["PREC"].values,
            hovertext=tooltips,
            hoverlabel=HOVERTOOL_FONT_FACE,
            hoverinfo="text",
            marker=dict(color=line_color),
            name=f"{line_name} AUC: {auc_value:0.4f}",
            showlegend=True,
        )
    )

    fig.update_layout(
        autosize=False,
        title="Precision Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=fig_size[0],
        height=fig_size[1],
    )

    return fig
