from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def metrics_df(labels: List[int], probas: List[float]) -> pd.DataFrame:
    """Create a metrics dataframe for binary classification problems

    Parameters
    ----------
    labels : List[int]
        List of the labels. Expected to be 0s and 1s
    probas : List[float]
        List of the probabilities.

    Returns
    -------
    pd.DataFrame with columns as:

        THRESHOLD : float
        TP: int, True Positives
        FP: int, False Positives
        FN: int, False Negatives
        TN: int, True Negatives
        FPR: float, False Positive Rate
        REC: float, Recall (or True Positive Rate)
        PREC: float, Precision
    """

    fprs, tprs, ths = roc_curve(labels, probas)
    _, (N, P) = np.unique(labels, return_counts=True)
    N = int(N)
    P = int(P)
    FPs = [int(fpr * N) for fpr in fprs]
    TPs = [int(tpr * P) for tpr in tprs]
    FNs = [P - TP for TP in TPs]
    TNs = [N - FP for FP in FPs]
    df = pd.DataFrame(
        zip(ths, TPs, FPs, FNs, TNs, fprs, tprs),
        columns=["THRESHOLD", "TP", "FP", "FN", "TN", "FPR", "REC"],
    )
    df["PREC"] = df[["TP", "FP"]].apply(
        lambda row: row["TP"] / (row["TP"] + row["FP"]), axis=1
    )

    return df


def cm_table(
    row: pd.Series,
    threshold: Optional[float] = None,
    line_break="<br>",
    pos_label="POS",
    neg_label="NEG",
) -> str:
    """Autoformats a confusion matrix table and includes some metrics. Used in
    the tooltip for precision recall/ROC curves

    Parameters
    ----------
    row : pd.Series
        pandas series with index containing
        * TP , FN, FP, TN (required)
        * any other indices will be appended at the top
    line_break : str, optional
        str to use for line breaks, by default "<br>". Use "\n" if using print()
    pos_label : str, optional
        Descrition of the positive label, by default "POS"
    neg_label : str, optional
        Description of the negative label, by default "NEG"

    Returns
    -------
    str
        A string formatted with metrics and the confusion matrix
    """

    cm = [row["TP"], row["FP"], row["FN"], row["TN"]]

    cm = [str(ii) for ii in cm]
    row_cell_sz = max([len(pos_label), len(neg_label)]) + 1
    cell_sz = max([len(ii) for ii in cm] + [row_cell_sz]) + 1
    cm_text = [ii.center(cell_sz) for ii in cm]

    col_pos = pos_label.center(cell_sz)
    col_neg = neg_label.center(cell_sz)
    col_sep = " " * row_cell_sz

    row_pos = pos_label.center(row_cell_sz)
    row_neg = neg_label.center(row_cell_sz)
    row_sep = "-" * row_cell_sz

    # cell separator
    cell_sep = "-" * cell_sz

    other_info_cols = [x for x in row.index if x not in {"TP", "TN", "FP", "FN"}]
    other_info_str = format_row(row, other_info_cols, line_break=line_break)
    # fmt: off
    table = (
        other_info_str
        + f"      {col_sep}|{'Actual'.center(cell_sz*2+1)}"     + f"|{line_break}"
        + f"      {col_sep}|{col_pos}"      + f"|{col_neg}"     + f"+{line_break}"
        + f"      {row_sep}+{cell_sep}"     + f"+{cell_sep}"    + f"+{line_break}"
        + f"      {row_pos}|{cm_text[0]}"   + f"|{cm_text[1]}"  + f"|{line_break}"
        + f"PRED  {row_sep}+{cell_sep}"     + f"+{cell_sep}"    + f"+{line_break}"
        + f"      {row_neg}|{cm_text[2]}"   + f"|{cm_text[3]}"  + f"|{line_break}"
        + f"      {row_sep}+{cell_sep}"     + f"+{cell_sep}"    + f"+{line_break}"
    )
    # fmt: on

    return table


def format_row(row: pd.Series, idxs: List[str], line_break="<br>") -> str:
    """Formats rows for prettier tooltip outputs"""
    cell_sz = max([len(idx) for idx in idxs])
    lines = [
        f"{idx.ljust(cell_sz)}: {row[idx]:0.3f}"
        if isinstance(row[idx], float)
        else f"{idx.ljust(cell_sz)}: {row[idx]}"
        for idx in idxs
    ]
    return line_break.join(lines) + line_break * 2
