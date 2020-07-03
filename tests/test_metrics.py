import numpy as np

from plotly_roc import metrics

labels = [0, 0, 0, 0, 1, 1, 1, 1]
probas = [0, 0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.9]


def test_metrics_df():
    metrics_df = metrics.metrics_df(labels, probas)
    expected_columns = ["THRESOLD", "TP", "FP", "FN", "TN", "PREC", "REC", "FPR"]
    expected_ths = np.array([1.9, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0])

    expected_tp = [0, 1, 2, 2, 3, 3, 4, 4]
    expected_fp = [0, 0, 0, 1, 1, 2, 2, 4]
    expected_fn = [4, 3, 2, 2, 1, 1, 0, 0]
    expected_tn = [4, 4, 4, 3, 3, 2, 2, 0]
    expected_fpr = [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 1.0]
    expected_rec = [0.0, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
    expected_prec = [nan, 1.0, 1.0, 0.66666667, 0.75, 0.6, 0.66666667, 0.5]

    assert all(col in metrics_df.columns for col in expected_columns)
