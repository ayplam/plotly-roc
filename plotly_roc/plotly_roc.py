import plotly.graph_objects as go
from sklearn.metrics import auc
from .metrics import metrics_df, cm_table

HOVERTOOL_FONT_FACE = {"font": {"family": "Courier New, monospace"}}


def roc_curve(
    metrics_df,
    fig=None,
    line_name=None,
    line_color="green",
    cm_labels=None,
    fig_size=(600, 500),
):
    cm_kwargs = dict()
    if cm_labels is not None:
        cm_kwargs["neg_label"] = cm_labels[0]
        cm_kwargs["pos_label"] = cm_labels[1]

    tooltips = metrics_df.apply(
        lambda row: cm_table(
            [int(row["TP"]), int(row["FP"]), int(row["FN"]), int(row["TN"])],
            **cm_kwargs,
        ),
        axis=1,
    )

    if fig is None:
        fig = go.Figure()

    if line_name:
        line_name += ";"
    else:
        line_name = ""

    fig.add_trace(
        go.Scatter(
            x=metrics_df["FPR"].values,
            y=metrics_df["REC"].values,
            hovertext=tooltips,
            hoverlabel=HOVERTOOL_FONT_FACE,
            hoverinfo="text",
            marker=dict(color=line_color),
            name=f'{line_name} AUC: {"%0.4f" % auc(metrics_df["FPR"], metrics_df["REC"])}',
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
        autosize=False, width=fig_size[0], height=fig_size[1],
    )

    return fig


import plotly.graph_objects as go
from sklearn.metrics import auc


def prec_recall_curve(
    metrics_df,
    fig=None,
    line_name=None,
    line_color="green",
    cm_labels=None,
    fig_size=(600, 500),
):
    tooltips = metrics_df.apply(
        lambda row: cm_table(
            [int(row["TP"]), int(row["FP"]), int(row["FN"]), int(row["TN"])]
        ),
        axis=1,
    )

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

    fig.add_trace(
        go.Scatter(
            x=metrics_df["REC"].values,
            y=metrics_df["PREC"].values,
            hovertext=tooltips,
            hoverlabel=HOVERTOOL_FONT_FACE,
            hoverinfo="text",
            marker=dict(color=line_color),
            name=f'{line_name} AUC: {"%0.4f" % auc(metrics_df["FPR"], metrics_df["REC"])}',
        )
    )

    fig.update_layout(
        autosize=False, width=fig_size[0], height=fig_size[1],
    )

    return fig