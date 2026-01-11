import json
import shutil
import os
import sys

def compare_and_promote(xgboost_dir="artifacts/xgboost", tabnet_dir="artifacts/tabnet", output_dir="artifacts"):
    """
    Selects the best model and promotes it to the root artifacts directory
    where the API expects to find it.
    """
    print("Starting Model Evaluation...")

    xgb_metrics_path = os.path.join(xgboost_dir, "validation_metrics.json")
    tab_metrics_path = os.path.join(tabnet_dir, "validation_metrics.json")
    
    # Default to XGBoost if comparison fails
    winner_dir = xgboost_dir
    winner_name = "xgboost"

    if os.path.exists(xgb_metrics_path) and os.path.exists(tab_metrics_path):
        with open(xgb_metrics_path) as f: xgb_score = json.load(f)["roc_auc"]
        with open(tab_metrics_path) as f: tab_score = json.load(f)["roc_auc"]
        
        print(f"   XGBoost ROC-AUC: {xgb_score:.4f}")
        print(f"   TabNet  ROC-AUC: {tab_score:.4f}")

        if tab_score > xgb_score:
            winner_dir = tabnet_dir
            winner_name = "tabnet"
    else:
        print("!!! Metrics missing. Defaulting to XGBoost.")

    print(f"WINNER!: {winner_name.upper()}")

    # Clear root artifacts folder (except subfolders to avoid errors)
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)

    # Copy winner artifacts to root "artifacts/"
    for item in os.listdir(winner_dir):
        s = os.path.join(winner_dir, item)
        d = os.path.join(output_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)

    # Save evaluation summary
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump({"winner": winner_name}, f)

if __name__ == "__main__":
    compare_and_promote()