# Scrap Sorting ML System
Real-time classification of recyclable materials (cardboard, glass, metal, paper, plastic, trash) using computer vision and deep learning.  

This project demonstrates how transfer learning and lightweight deployment can enable automated scrap sorting for industrial recycling systems.

---

## Motivation
Recycling and waste management require accurate separation of materials. Manual sorting is:
- Labor-intensive
- Error-prone
- Expensive at scale  

Using **deep learning**, we can build automated systems capable of identifying and sorting scrap in real time, reducing cost and increasing efficiency.

---

## Key Features
- **6-Class Classification**: Cardboard, Glass, Metal, Paper, Plastic, Trash  
- **Data Augmentation**: Improves generalization on limited datasets  
- **Transfer Learning**: ResNet18 pretrained on ImageNet, fine-tuned on TrashNet dataset  
- **Lightweight Deployment**: Exported to both **TorchScript** and **ONNX** for edge devices  
- **Conveyor Simulation**: Mimics real-time scrap sorting pipeline  
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix  

---

## Dataset
- **Source**: [TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet)  
- **Classes**: `cardboard, glass, metal, paper, plastic, trash`  
- **Split**: 70% train, 20% validation, 10% test  
- **Augmentations**: Random rotation, flips, color jittering  

---

## Model & Training
- **Architecture**: ResNet18  
- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam (lr = 1e-4)  
- **Scheduler**: ReduceLROnPlateau  
- **Training**: 3 epochs on Colab GPU (T4)  

### Exports:
- PyTorch checkpoint (`.pth`)  
- TorchScript (`.pt`)  
- ONNX (`.onnx`)  

---

## 📈 Results

### Classification Report
```

```
          precision    recall  f1-score   support
```

cardboard     0.9740    1.0000    0.9868        75
glass     0.9091    0.8738    0.8911       103
metal     0.8462    0.8462    0.8462        78
paper     0.9569    0.9737    0.9652       114
plastic     0.9048    0.9048    0.9048       105
trash     0.8462    0.8462    0.8462        26

```
accuracy                         0.9162       501
```

macro avg     0.9062    0.9074    0.9067       501
weighted avg     0.9157    0.9162    0.9158       501

```

- **Overall Accuracy**: ~91.6%  
- **Strong performance** on cardboard, paper, and plastic.  
- **Slightly lower recall** for glass, metal, and trash, due to class imbalance in dataset.  

### Confusion Matrix
Confusion matrix is available in `/results/confusion_matrix.png`.  

---

## 📂 Project Structure
```

scrap_sorting_project/
│
├── src/                      # Source code, training & inference scripts
│   ├── training_notebook.ipynb
│   ├── inference_colab.py
│   └── inference_jetson.py
│
├── data/                     # Sample test data
│   ├── sample_test.jpg
│   └── README.md (dataset link)
│
├── models/                   # Saved models
│   ├── resnet18_best.pth
│   ├── resnet18_scrap.onnx
│   └── resnet18_scrap_scripted.pt
│
├── results/                  # Outputs and visualizations
│   ├── predictions.csv
│   └── confusion_matrix.png
│
├── README.md                 # Documentation
└── performance_report.md     # Visual summary of results

````

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install torch torchvision datasets huggingface_hub tqdm scikit-learn onnxruntime pillow matplotlib
````

### 2. Train Model (optional, skip if using provided weights)

```bash
jupyter notebook src/training_notebook.ipynb
```

### 3. Run Inference

```bash
python src/inference_colab.py
```

### 4. Conveyor Simulation

```bash
python src/inference_colab.py --simulate
```

### 5. Check Results

* CSV logs → `/results/predictions.csv`
* Confusion Matrix → `/results/confusion_matrix.png`

---

## Performance Report

See `performance_report.md` for:

* Training curves
* Confusion matrix
* Per-class precision/recall
* Key takeaways

---

## 🔧 Future Work

* Deploy on Jetson Nano / Xavier for edge inference
* Add manual override logic for misclassifications
* Active learning loop: retrain with misclassified samples

---

## 🙌 Credits

* Dataset: [garythung/TrashNet](https://huggingface.co/datasets/garythung/trashnet)
* Frameworks: PyTorch, Hugging Face Datasets, ONNX Runtime

```
