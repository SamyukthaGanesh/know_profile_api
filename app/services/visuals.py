import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

def chart_top_spend(top_spend: pd.Series) -> str:
    fig = plt.figure()
    top_spend.plot(kind="bar")
    plt.title("Top Spend Categories (INR)")
    plt.xlabel("Category")
    plt.ylabel("Amount (INR)")
    plt.tight_layout()
    path = ASSETS / "top_spend.png"
    plt.savefig(path)
    plt.close(fig)
    return str(path)

def chart_dti(dti: float) -> str:
    fig = plt.figure()
    plt.bar(["DTI"], [dti])
    plt.ylim(0, 1)
    plt.title("Debt-to-Income Ratio")
    plt.tight_layout()
    path = ASSETS / "dti_ratio.png"
    plt.savefig(path)
    plt.close(fig)
    return str(path)

def chart_importance(importance: dict) -> str:
    # importance is dict name -> float
    # plot top 12
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:12]
    names = [k for k,_ in items][::-1]
    vals = [v for _,v in items][::-1]
    fig = plt.figure()
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), names)
    plt.title("Global Feature Importance (proxy)")
    plt.xlabel("Importance (|weight|)")
    plt.tight_layout()
    path = ASSETS / "feature_importance.png"
    plt.savefig(path)
    plt.close(fig)
    return str(path)
