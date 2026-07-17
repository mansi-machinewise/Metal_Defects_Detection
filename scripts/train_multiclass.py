# scripts/train_multiclass.py
from ultralytics import YOLO
from pathlib import Path
import yaml
import torch

def main():
    project_root = Path(__file__).parents[1].resolve()
    data_yaml    = project_root / "dataset_multiclass" / "data.yaml"

    # Auto-update path for current machine
    content = yaml.safe_load(data_yaml.read_text())
    content["path"] = (project_root / "dataset_multiclass").as_posix()
    data_yaml.write_text(yaml.dump(content))

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU : {gpu_name}")
        print(f"VRAM: {vram_gb:.1f} GB")
        device = "0"
    else:
        print("No GPU found — training on CPU")
        device = "cpu"

    model = YOLO("yolov8m.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=4,
        lr0=0.01,
        patience=20,
        device=device,
        workers=2,           # reduced to avoid RAM pressure
        project=str(project_root / "outputs" / "runs"),
        name="defect_multiclass_v1",
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.1,
        degrees=10.0,
        cache=False,         # disabled — not enough free RAM
    )

    print(f"\nTraining complete")
    print(f"Best weights: {project_root}/outputs/runs/defect_multiclass_v1/weights/best.pt")

if __name__ == '__main__':
    main()