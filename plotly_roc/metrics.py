from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd


def metrics_df(labels, probas):
    """Gets metrics dataframe. For each threshold, returns
    THRESHOLD
    TP, True Positives
    FP, False Positives
    FN, False Negatives
    TN, True Negatives
    FPR, False Positive Rate
    REC, Recall (or True Positive Rate)
    PREC, Precision
    
    
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


def cm_table(cm, line_break="<br>", pos_label="POS", neg_label="NEG"):
    """Makes a confusion matrix table

    """

    prec = cm[0] / (cm[0] + cm[1]) if cm[0] else 0
    recall = cm[0] / (cm[0] + cm[2]) if cm[0] else 0

    cm = [str(ii) for ii in cm]
    row_cell_sz = max([len(pos_label), len(neg_label)]) + 1
    cell_sz = max([len(ii) for ii in cm] + [row_cell_sz]) + 1
    cm_text = [ii.center(cell_sz) for ii in cm]

    col_pos = pos_label.center(cell_sz)
    col_neg = neg_label.center(cell_sz)
    col_sep = " "*row_cell_sz

    row_pos = pos_label.center(row_cell_sz)
    row_neg = neg_label.center(row_cell_sz)
    row_sep = "-"*row_cell_sz
    

    # cell separator
    cell_sep = "-" * cell_sz

    # fmt: off
    out = (
        f"PRECISION: {'%0.3f' % prec} {line_break}"                            +
        f"RECALL   : {'%0.3f' % recall} {line_break}{line_break}"              +
        f"      {col_sep}|{'Actual'.center(cell_sz*2+1)}"    +f"|{line_break}" +
        f"      {col_sep}|{col_pos}"      +f"|{col_neg}"     +f"+{line_break}" +
        f"      {row_sep}+{cell_sep}"     +f"+{cell_sep}"    +f"+{line_break}" +
        f"      {row_pos}|{cm_text[0]}"   +f"|{cm_text[1]}"  +f"|{line_break}" +
        f"PRED  {row_sep}+{cell_sep}"     +f"+{cell_sep}"    +f"+{line_break}" +       
        f"      {row_neg}|{cm_text[2]}"   +f"|{cm_text[3]}"  +f"|{line_break}" +
        f"      {row_sep}+{cell_sep}"     +f"+{cell_sep}"    +f"+{line_break}" 
    )
    # fmt: on

    return out
