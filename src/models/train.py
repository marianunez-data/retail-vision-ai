"""YOLO11n baseline training for RetailVision AI."""
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# Disable MLflow to save RAM
os.environ["MLFLOW_TRACKING_URI"] = ""

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Load config
with open(PROJECT_ROOT / "configs" / "base_config.yaml") as f:
    config = yaml.safe_load(f)

def main():
    model = YOLO(config["training"]["model"])
    model.train(
        data=str(PROCESSED / "data.yaml"),
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["image_size"],
        batch=config["training"]["batch_size"],
        patience=config["training"]["patience"],
        project=str(PROJECT_ROOT / "models"),
        name="yolo11n_baseline_v1",
        exist_ok=True,
        device=0,
        workers=0,
        verbose=True,
    )

if __name__ == "__main__":
    main()
