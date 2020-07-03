import pandas as pd

from plotly_roc import metrics

labels = [0, 0, 0, 0, 1, 1, 1, 1]
probas = [0, 0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.9]


def test_metrics_df():
    metrics_df = metrics.metrics_df(labels, probas)

    expected_row = {
        "THRESHOLD": 0.5,
        "TP": 3.0,
        "FP": 1,
        "FN": 1,
        "TN": 3,
        "FPR": 0.25,
        "REC": 0.75,
        "PREC": 0.75,
    }

    assert all([col in metrics_df.columns for col in expected_row])

    row = metrics_df.iloc[4].to_dict()
    for kk in row:
        assert row[kk] == expected_row[kk]


def test_cm_table():
    sample_row = pd.Series(
        {
            "THRESHOLD": 0.12345,
            "TP": 99999,
            "FP": 1,
            "FN": 1,
            "TN": 99999,
            "FPR": 0.25,
            "REC": 0.75,
            "PREC": 0.75,
            "RANDOM": "Hello world",
        }
    )
    table = metrics.cm_table(sample_row, line_break="\n")
    rows = table.split("\n")

    # Ensure the colons line up
    header = rows[:5]
    assert all([h[9] == ":" for h in header])

    # Ensure the verticals in the table align
    confusion_matrix = rows[8:]
    assert all([row[10] in ["|", "+"] for row in confusion_matrix if row])
    assert all([row[17] in ["|", "+"] for row in confusion_matrix if row])
    assert all([row[-1] in ["|", "+"] for row in confusion_matrix if row])
