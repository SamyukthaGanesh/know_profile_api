import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float,float]:
    tp = float(((y_true==1)&(y_pred==1)).sum())
    tn = float(((y_true==0)&(y_pred==0)).sum())
    fp = float(((y_true==0)&(y_pred==1)).sum())
    fn = float(((y_true==1)&(y_pred==0)).sum())
    tpr = tp / (tp+fn+1e-9)
    fpr = fp / (fp+tn+1e-9)
    pos_rate = float((y_pred==1).mean())
    return tpr, fpr, pos_rate, float(tp+tn)/len(y_true)

def statistical_parity(y_pred: np.ndarray, groups: np.ndarray) -> float:
    uniq = np.unique(groups)
    if len(uniq) < 2: return 0.0
    rates = [float(np.mean(y_pred[groups==g]==1)) for g in uniq]
    return max(rates) - min(rates)

def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    uniq = np.unique(groups)
    tprs = []
    for g in uniq:
        mask = groups==g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt==1)&(yp==1)).sum(); fn = ((yt==1)&(yp==0)).sum()
        tprs.append(float(tp/(tp+fn+1e-9)))
    return max(tprs) - min(tprs)

def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str,float]:
    uniq = np.unique(groups)
    tprs, fprs = [], []
    for g in uniq:
        mask = groups==g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt==1)&(yp==1)).sum(); fn = ((yt==1)&(yp==0)).sum()
        fp = ((yt==0)&(yp==1)).sum(); tn = ((yt==0)&(yp==0)).sum()
        tpr = float(tp/(tp+fn+1e-9)); fpr = float(fp/(fp+tn+1e-9))
        tprs.append(tpr); fprs.append(fpr)
    return {"tpr_gap": max(tprs)-min(tprs), "fpr_gap": max(fprs)-min(fprs)}
