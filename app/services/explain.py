from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict

# Optional SHAP/LIME import; we gracefully degrade if not available
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

def nlg_from_importances(imp: Dict[str, float], Xrow: pd.Series, top_k: int = 3) -> List[str]:
    # pick top-k features by absolute importance and craft human sentences
    ranked = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    lines = []
    for feat, _ in ranked:
        val = Xrow.get(feat, None)
        if val is None:
            continue
        # simple templates
        if "credit_score" in feat:
            lines.append(f"Your credit score ({int(val)}) influenced the decision positively.")
        elif "dti_ratio" in feat:
            lines.append(f"A debt-to-income ratio of {val:.2f} affected approval odds.")
        elif "Shopping" in feat or "Dining" in feat or "Groceries" in feat:
            lines.append(f"Spending in {feat} contributed to the outcome (amount: ₹{int(val)}).")
        elif "annual_interest_inr" in feat:
            lines.append(f"Interest income (₹{int(val)}) was considered for stability.")
        else:
            lines.append(f"Feature '{feat}' with value {val:.2f} was influential.")
    if not lines:
        lines = ["Stable payment history and spending patterns supported the decision."]
    return lines

def groupwise_shap_parity(shap_matrix: np.ndarray, groups: List[str]) -> float:
    # Measure disparity between group mean |SHAP| vectors via cosine distance
    # This is a proxy for "reason parity" (lower is better)
    import numpy as np
    uniq = sorted(set(groups))
    if len(uniq) < 2 or shap_matrix.size == 0:
        return 0.0
    means = []
    for g in uniq:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        means.append(np.mean(np.abs(shap_matrix[idx]), axis=0))
    if len(means) < 2:
        return 0.0
    a, b = means[0], means[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    sim = float(np.dot(a, b) / denom)
    return float(1.0 - sim)  # distance
