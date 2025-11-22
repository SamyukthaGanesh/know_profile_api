from typing import Dict, List
import numpy as np

def bias_reduction_report(history: Dict[str, List[float]]) -> Dict:
    # history: metric -> list over time (months)
    notes = []
    for k, seq in history.items():
        if len(seq) >= 2 and seq[-1] < seq[0]:
            notes.append(f"{k} improved from {seq[0]:.3f} to {seq[-1]:.3f}.")
        elif len(seq) >= 2:
            notes.append(f"{k} changed from {seq[0]:.3f} to {seq[-1]:.3f}.")
    return {"metrics": history, "notes": notes}
