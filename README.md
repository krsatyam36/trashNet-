<div align="center">

# TrashNet — Automated Waste Classification

**v1.0.0** — *A deep learning-based waste sorting system using ResNet18, built for smart recycling automation*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Export-005CED?style=flat&logo=onnx&logoColor=white)](https://onnx.ai)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Hugging Face](https://img.shields.io/badge/Datasets-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/garythung/trashnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.23%25-success)](performace/performance_report)

**Created by [Kumar Satyam](mailto:kumarsatyam3135@gmail.com)**

Classifies waste into **6 categories** — cardboard, glass, metal, paper, plastic, trash — achieving **86.23% test accuracy** with a ResNet18 model fine-tuned via transfer learning. Includes a simulated real-time conveyor belt sorting pipeline, TorchScript/ONNX model export, and comprehensive evaluation.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Performance](#model-performance)
- [Conveyor Simulation](#conveyor-simulation)
- [Model Export](#model-export)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

```mermaid
graph TB
    subgraph Input
        A[TrashNet Dataset<br/>5,054 Images] --> B[Train/Val/Test Split<br/>70/20/10%]
    end

    subgraph Preprocessing
        B --> C[Resize 224×224]
        C --> D[Data Augmentation<br/>Flip, Rotation, ColorJitter]
        D --> E[Normalization<br/>ImageNet mean/std]
    end

    subgraph Training
        E --> F[ResNet18 Backbone<br/>ImageNet Pretrained]
        F --> G[Fine-tuned Classifier<br/>6 output classes]
        G --> H{Validation<br/>Accuracy}
        H -->|improves| I[Save Best<br/>Checkpoint]
        H -->|no improvement| J[Reduce LR<br/>on Plateau]
    end

    subgraph Export
        I --> K[TorchScript<br/>resnet18_scrap_scripted.pt]
        I --> L[ONNX<br/>resnet18_scrap.onnx]
    end

    subgraph Inference
        K --> M[Conveyor Simulation<br/>100 frames, 0.2s interval]
        L --> M
        M --> N[Predictions CSV]
        M --> O[10×10 Grid Viz]
    end

    subgraph Evaluation
        I --> P[Test Set Inference<br/>501 images]
        P --> Q[Classification Report]
        P --> R[Confusion Matrix]
    end
```

---

## Architecture

### Model: ResNet18 with Transfer Learning

```mermaid
flowchart LR
    A[Input Image<br/>224×224×3] --> B[ResNet18<br/>Convolutional Backbone<br/>ImageNet Pretrained]
    B --> C[Global Average<br/>Pooling]
    C --> D[Fully Connected<br/>512 → 6]
    D --> E[Softmax]
    E --> F[Cardboard]
    E --> G[Glass]
    E --> H[Metal]
    E --> I[Paper]
    E --> J[Plastic]
    E --> K[Trash]
```

| Component | Detail |
|-----------|--------|
| **Backbone** | ResNet18 pre-trained on ImageNet |
| **Modification** | Final FC layer replaced: `512 → 6` |
| **Input Size** | 224×224 pixels, RGB |
| **Loss** | CrossEntropyLoss |
| **Optimizer** | Adam (lr = 1×10⁻⁴) |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=2) |
| **Epochs** | 3 |
| **Batch Size** | 16 |

---

## Tech Stack

```mermaid
mindmap
  root((TrashNet))
    Framework
      PyTorch
      TorchVision
      TorchScript
      ONNX / ONNX Runtime
    Data
      Hugging Face Datasets
      NumPy
      Pillow
    Training
      Google Colab A100
      Adam Optimizer
      ReduceLROnPlateau
      tqdm
    Evaluation
      scikit-learn
      Matplotlib
      Classification Report
      Confusion Matrix
    Export
      TorchScript JIT
      ONNX opset 11
      Dynamic Batching
```

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework for model building and training |
| **TorchVision** | Pre-trained ResNet18 models and image transforms |
| **Hugging Face Datasets** | Dataset loading (`garythung/trashnet`) |
| **ONNX / ONNX Runtime** | Cross-platform model export and inference |
| **scikit-learn** | Classification metrics and confusion matrix |
| **Matplotlib** | Visualization (confusion matrix, conveyor grid) |
| **Google Colab** | GPU-accelerated training (NVIDIA A100) |

---

## Dataset

The **TrashNet** dataset ([garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet)) contains **5,054** images of waste items across six classes:

| Class      | Description | Train | Val | Test |
|------------|-------------|-------|-----|------|
| Cardboard  | Corrugated cardboard / paperboard | ~525 | ~150 | 75 |
| Glass      | Bottles, jars, glass items | ~721 | ~206 | 103 |
| Metal      | Cans, aluminum, metal objects | ~546 | ~156 | 78 |
| Paper      | Paper, newspaper, magazines | ~798 | ~228 | 114 |
| Plastic    | Plastic bottles, containers, bags | ~735 | ~210 | 105 |
| Trash      | Miscellaneous non-recyclable waste | ~182 | ~52 | 26 |
| **Total**  | | **3,537** | **1,016** | **501** |

### Data Augmentation (Training Only)

```mermaid
flowchart LR
    A[Original Image] --> B[Random Horizontal Flip]
    A --> C[Random Rotation ±10°]
    A --> D[Color Jitter<br/>brightness/contrast/saturation]
    B & C & D --> E[Resize 224×224]
    E --> F[Normalize<br/>ImageNet mean/std]
    F --> G[Augmented Batch]
```

- `RandomHorizontalFlip` — mirror images with 50% probability
- `RandomRotation(10)` — small rotational variations
- `ColorJitter(0.2, 0.2, 0.2, 0.05)` — brightness, contrast, saturation shifts

---

## Pipeline

### Full Training & Evaluation Pipeline

```mermaid
flowchart TD
    START([Start]) --> SETUP[Setup: Mount Drive<br/>HF Login, Create Dirs]
    SETUP --> LOAD[Load Dataset<br/>Hugging Face: garythung/trashnet]
    LOAD --> SPLIT[Train/Val/Test Split<br/>70/20/10% with seed=42]
    SPLIT --> PREPROC[Preprocess & Augment<br/>224×224, Normalize, Random Transforms]
    PREPROC --> DATALOADER[Create DataLoaders<br/>Batch=16, Workers=2]
    DATALOADER --> TRAIN[Train ResNet18<br/>Adam, CrossEntropyLoss, 3 epochs]
    TRAIN --> CHECK{Val Accuracy<br/>Improved?}
    CHECK -->|Yes| SAVE[Save Best Checkpoint<br/>resnet18_best.pth]
    CHECK -->|No| SCHED[Reduce LR]
    SAVE --> EXPORT[Export Models]
    SCHED --> TRAIN
    EXPORT --> TORCH[TorchScript<br/>resnet18_scrap_scripted.pt]
    EXPORT --> ONNX[ONNX<br/>resnet18_scrap.onnx]
    TORCH --> TEST[Test Inference<br/>& Verify Consistency]
    ONNX --> TEST
    TEST --> CONVEY[Conveyor Simulation<br/>100 Sequential Frames]
    CONVEY --> CSV[Generate Predictions CSV]
    CONVEY --> GRID[Generate 10×10 Grid]
    TEST --> EVAL[Full Test Evaluation<br/>501 Images]
    EVAL --> CLASS_REPORT[Classification Report]
    EVAL --> CONF_MATRIX[Confusion Matrix]
    CONF_MATRIX --> END([End])
```

---

## Model Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **86.23%** |
| Precision (macro avg) | 84.35% |
| Recall (macro avg) | 84.80% |
| F1-score (macro avg) | 84.32% |

### Per-Class Performance

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Cardboard  | 0.9412    | 0.8533 | 0.8951   | 75      |
| Glass      | 0.9072    | 0.8544 | 0.8800   | 103     |
| Metal      | 0.7526    | 0.9359 | 0.8343   | 78      |
| Paper      | 0.8947    | 0.8947 | 0.8947   | 114     |
| Plastic    | 0.8866    | 0.8190 | 0.8515   | 105     |
| Trash      | 0.6786    | 0.7308 | 0.7037   | 26      |

### Confusion Matrix

![Normalized Confusion Matrix](data/confusion_matrix.png)

**Key Observations:**
- **Best performing**: Cardboard (94% precision), Metal (94% recall)
- **Worst performing**: Trash class — only 26 test samples, visually diverse, often confused with plastic and paper
- ~8% of predictions fell below the 60% confidence threshold

---

## Conveyor Simulation

A real-time waste-sorting simulation that processes 100 test images sequentially (0.2s per frame), mimicking a smart recycling facility's conveyor belt camera feed.

```mermaid
sequenceDiagram
    participant Camera
    participant Model as ResNet18
    participant Logger
    participant Viz as Grid Visualizer

    loop 100 Frames (0.2s interval)
        Camera->>Model: Frame N (224×224 RGB)
        Model->>Model: Inference
        Model-->>Logger: Predicted Class + Confidence
        Logger->>Logger: Log to CSV<br/>(frame_id, class, conf, timestamp)
        Logger->>Logger: Flag if conf < 60%
        Model-->>Viz: Image + Prediction<br/>for grid display
        Note over Camera,Model: 0.2 second delay
    end

    Logger->>Logger: Save predictions.csv
    Viz->>Viz: Render 10×10 Grid
    Viz->>Viz: Save conveyor_grid_100.png
```

### Sample Predictions

| Frame | Predicted | Confidence | Flag |
|-------|-----------|------------|------|
| 000   | trash     | 94.90%     | ✅   |
| 002   | glass     | 59.13%     | ⚠️ Low |
| 004   | paper     | 99.89%     | ✅   |
| 016   | cardboard | 99.88%     | ✅   |
| 087   | trash     | 58.56%     | ⚠️ Low |

![Conveyor Grid](data/conveyor_grid_100.png)

### Configuration

| Parameter | Value |
|-----------|-------|
| Confidence threshold | 60% |
| Frame interval | 0.2–0.3 seconds |
| Max frames | 100 |
| Inference backend | TorchScript (default) / ONNX |

---

## Model Export

Models are exported for production deployment in two formats:

### TorchScript
```python
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(checkpoint["model_state"])
scripted = torch.jit.trace(model, example_input)
scripted.save("resnet18_scrap_scripted.pt")
```

### ONNX
```python
torch.onnx.export(model, example_input, "resnet18_scrap.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11)
```

### Download Pre-trained Models

Model files are too large for GitHub. Download from Google Drive:

| Model | Format | Link |
|-------|--------|------|
| Best Checkpoint | `.pth` | [Download](https://drive.google.com/file/d/10JszPCeLGit6z2AhdHYiCopa_UErfrFH/view?usp=sharing) |
| TorchScript | `.pt` | [Download](https://drive.google.com/file/d/1yxiXjOR5bKCjUFg6Je_xaWrPj3rEsgYa/view?usp=sharing) |
| ONNX | `.onnx` | [Download](https://drive.google.com/file/d/1Oa7X3O249ja09acwTNWVQW8wQjy9E2Ne/view?usp=sharing) |

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision datasets huggingface_hub tqdm scikit-learn onnx onnxruntime pillow matplotlib numpy
```

### Run the Full Pipeline

```bash
# Clone the repo
git clone https://github.com/krsatyam36/trashNet-.git
cd trashNet-

# Run the complete training + evaluation script
python src/trash_net_script.py
```

> **Note**: The script was designed for Google Colab and includes Colab-specific commands (`drive.mount`, `!pip`, etc.). For local execution, comment out Colab-specific cells and adjust paths accordingly.

### Run Inference on a Single Image

```python
import torch
from torchvision import transforms
from PIL import Image

# Load TorchScript model
model = torch.jit.load("models/resnet18_scrap_scripted.pt").eval()

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Predict
img = Image.open("your_image.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)
with torch.no_grad():
    probs = torch.nn.functional.softmax(model(x), dim=1)

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
pred_idx = probs.argmax().item()
print(f"Predicted: {classes[pred_idx]} ({probs[0][pred_idx]:.2%})")
```

---

## Repository Structure

```
trashNet-/
├── README.md                          # Project documentation (this file)
├── LICENSE                            # MIT License
├── .gitignore
├── data/
│   ├── predictions.csv                # 100-frame conveyor simulation output
│   ├── confusion_matrix.png           # Normalized confusion matrix
│   ├── conveyor_grid_100.png          # 10×10 prediction grid visualization
│   └── output_csv/                    # Placeholder for additional CSVs
├── models/
│   └── model_info.txt                 # Google Drive links for model files
├── performace/
│   └── performance_report             # Detailed performance analysis report
└── src/
    ├── trash_net_script.py            # Complete Python pipeline script
    ├── trash_net_script_github.ipynb  # Full Jupyter notebook with outputs
    └── trash_net_script.ipynb         # Placeholder (refer to GitHub version)
```

---

## Results

### Visual Evaluation Grid

![Conveyor Grid](data/conveyor_grid_100.png)

The 10×10 grid shows 100 test images with their predicted class and confidence score. Low-confidence predictions (<60%) are flagged.

### Performance Report

A detailed performance report covering model architecture, training setup, per-class metrics, confusion matrix analysis, and recommendations is available at [`performace/performance_report`](performace/performance_report).

---

## Future Work

```mermaid
gantt
    title Roadmap
    dateFormat  YYYY-MM-DD
    section Model Improvement
    Increase Training Epochs          :done, 2025-10-01, 2025-10-15
    Class Balancing for Trash         :active, 2025-10-15, 2025-11-01
    Experiment with EfficientNet/ViT  :2025-11-01, 2025-12-01

    section Deployment
    Model Quantization (INT8)         :2025-11-01, 2025-11-15
    Edge Deployment (Raspberry Pi)     :2025-11-15, 2025-12-15
    Real-time Camera Integration       :2025-12-01, 2026-01-15

    section Production
    Confidence-based Filtering        :2025-11-01, 2025-11-15
    Web Dashboard                     :2025-12-15, 2026-02-01
```

- **Increase training epochs** — current 3-epoch model shows room for improvement
- **Class balancing** — the Trash class (only 26 test samples) needs more diverse data
- **Architecture exploration** — EfficientNet, MobileNet, or Vision Transformers for higher accuracy or edge deployment
- **Model quantization** — INT8 quantization for faster edge inference
- **Real-world lighting** — incorporate varied backgrounds and lighting conditions
- **Confidence-based filtering** — flag uncertain predictions during live sorting

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 newton4th
