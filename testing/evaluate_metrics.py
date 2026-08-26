# scripts/evaluate_metrics.py
"""
Proper evaluation with precision, recall, mAP metrics.
Runs YOLOv8's built-in validation which gives real metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

from ultralytics import YOLO
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

def main():
    cfg = load_config()
    model_path = cfg["inference"]["model_path"]
    data_yaml  = cfg["dataset"]["yaml"]
    
    logger.info("Running proper validation with mAP metrics...")
    logger.info("Model: %s", model_path)
    
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        augment=True,
        imgsz=640,
        batch=4,
        conf=0.25,
        iou=0.5,
        device="cpu",
        plots=True,
        save_json=True,
        project="outputs/reports",
        name="validation_metrics",
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    print(f"  Precision (P)    : {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall    (R)    : {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"  mAP@0.5          : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95     : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  F1 Score         : {2 * results.results_dict.get('metrics/precision(B)', 0) * results.results_dict.get('metrics/recall(B)', 0) / max(results.results_dict.get('metrics/precision(B)', 0) + results.results_dict.get('metrics/recall(B)', 0), 1e-6):.4f}")
    print("=" * 60)
    print()
    print("Plots saved to: outputs/reports/validation_metrics/")
    print("  - confusion_matrix.png")
    print("  - PR_curve.png")
    print("  - F1_curve.png")

if __name__ == "__main__":
    main()