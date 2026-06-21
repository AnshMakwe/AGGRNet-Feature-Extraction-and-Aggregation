# AGGRNet: Feature Extraction and Aggregation for Enhanced Medical Image Classification

AGGRNet augments the **YOLOv11 classification backbone** with two lightweight,
attention-based modules that explicitly separate *informative* from
*non-informative* features and then re-aggregate them through global
self-attention. The design targets fine-grained **medical image
classification** (e.g. gastrointestinal endoscopy), where discriminative cues
are often small, low-contrast, and spatially localized.

The entire contribution is implemented on top of
[Ultralytics](https://github.com/ultralytics/ultralytics) `v8.3.153` so that
existing YOLO training, validation, and export tooling can be reused without
modification.

---

## Method Overview

AGGRNet inserts a two-stage block at three resolution stages (P3, P4, P5) of the
YOLO11-cls backbone:

1. **Feature Extraction Module (FEM)** — `FeatureExtractionModule`
   - Computes a joint attention map from **spatial attention**
     (`spatial_attn_layer`) and **channel attention** (`ca_layer`).
   - Uses a **learnable soft threshold** to split the input into two streams:
     - `Xinfo`  — informative features
     - `Xninfo` — non-informative features
   - The threshold is applied with a temperature-scaled sigmoid
     `σ((a − τ) · T)` rather than a hard `≥` comparison, so the threshold `τ`
     stays differentiable and is learned end-to-end (a hard step has zero
     gradient almost everywhere and would freeze `τ`).
   - Output: the two streams concatenated along the channel axis (`2 × C`).

2. **Feature Aggregation Module (FAM)** — `FeatureAggregationModule`
   - Re-splits the `2 × C` tensor into `Xinfo` / `Xninfo`.
   - Applies multi-head **global attention** (`GlobalAttention`) with
     - Query `= Xinfo + Xninfo`
     - Key   `= Xinfo − Xninfo`
     - Value `= Xinfo`
   - Output: aggregated global features (`C` channels), which are concatenated
     with the corresponding earlier backbone stage before the next `C3k2` block.

All custom modules live in
[`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py) and are
registered in `ultralytics/nn/modules/__init__.py` and
`ultralytics/nn/tasks.py`. The architecture is defined in
[`ultralytics/cfg/models/11/yolo11-cls.yaml`](ultralytics/cfg/models/11/yolo11-cls.yaml).

---

## Repository Structure

```
.
├── Train.py                       # Train + validate the AGGRNet model
├── pred.py                        # Run inference / per-class evaluation, export to Excel
├── requirements.txt               # Python dependencies
├── ultralytics/                   # Ultralytics framework with AGGRNet modules
│   ├── nn/modules/block.py        # FEM, FAM, GlobalAttention, attention layers
│   └── cfg/models/11/yolo11-cls.yaml  # AGGRNet model definition
├── docker/ tests/                 # Upstream Ultralytics assets (unmodified)
└── LICENSE                        # AGPL-3.0 (inherited from Ultralytics)
```

---

## Installation

AGGRNet runs against the **bundled, modified `ultralytics/` package** in this
repository, *not* the PyPI release. Install the dependencies and run all scripts
from the repository root so the local package takes precedence.

```bash
# Python 3.9+ recommended
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

> Do **not** `pip install ultralytics`; doing so would shadow the local modified
> package. Always launch scripts from the repository root.

Verify the custom modules load correctly:

```bash
python -c "from ultralytics.nn.modules import FeatureExtractionModule, FeatureAggregationModule; print('AGGRNet modules OK')"
```

---

## Dataset

The model expects the standard **Ultralytics classification dataset layout**
(one subfolder per class):

```
dataset/
├── train/
│   ├── class_0/  *.jpg
│   ├── class_1/  *.jpg
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Set `nc` (number of classes) in
[`ultralytics/cfg/models/11/yolo11-cls.yaml`](ultralytics/cfg/models/11/yolo11-cls.yaml)
to match your dataset (default: `4`).

---

## Training

Edit the dataset path in [`Train.py`](Train.py):

```python
results = model.train(
    data="path/to/your/dataset",   # <-- set this
    epochs=100,
    imgsz=224,
    batch=16,
    device=0,
    ...
)
```

Then run:

```bash
python Train.py
```

This will:
- set all random seeds for reproducibility (`seed=42`),
- build the AGGRNet YOLO11-cls architecture and load pretrained weights
  (`yolo11l-cls.pt`, downloaded automatically; non-matching layers are skipped),
- train for 100 epochs on 224×224 images,
- save results under `runs/classify/<name>/`.

---

## Evaluation

Set the trained checkpoint and test folder in [`pred.py`](pred.py):

```python
model = YOLO("runs/classify/test_architecture/weights/best.pt")
test_folder = "path/to/your/dataset/test"
```

Then run:

```bash
python pred.py
```

This reports overall and per-class accuracy, runs standard YOLO validation
(Top-1 / Top-5), and exports per-image predictions to an Excel file.

---

## Configuration Reference

`yolo11-cls.yaml` exposes the AGGRNet modules as standard YOLO layers:

| Module                      | Args                                    |
|-----------------------------|-----------------------------------------|
| `FeatureExtractionModule`   | `[channels, kernel_size, threshold]`    |
| `FeatureAggregationModule`  | `[channels, num_heads, attn_ratio, sppf_k]` |

Model scales (`n`, `s`, `m`, `l`, `x`) follow the YOLO11 compound-scaling
convention; `Train.py` uses the `l` scale by default.

---

## Acknowledgements

Built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

## License

This project inherits the **AGPL-3.0** license from Ultralytics. See
[`LICENSE`](LICENSE) for details.
