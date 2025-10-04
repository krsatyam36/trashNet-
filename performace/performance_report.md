TrashNet Classification Model — Performance Report
1. Project Overview

This report summarizes the performance of a deep learning model trained on the TrashNet dataset for automated waste classification.
The model is based on ResNet18, fine-tuned using transfer learning, and tested on six waste categories: cardboard, glass, metal, paper, plastic, and trash.

The goal of this project was to build a reliable image classification system capable of recognizing and categorizing different types of recyclable and non-recyclable waste from real-world images.

2. Dataset and Experimental Setup

The dataset was loaded from Hugging Face (garythung/trashnet) and divided into three subsets:

Split	Percentage	Purpose
Training	70%	Model learning and optimization
Validation	20%	Hyperparameter tuning and early stopping
Test	10%	Final model evaluation

Each image was resized to 224x224 pixels and augmented with random rotations, flips, and color jitter to enhance robustness.
The model was trained on a GPU with three epochs due to compute constraints, and used Adam optimizer with a learning rate of 1e-4.

3. Model Performance
Overall Evaluation
Metric	Score
Accuracy	0.8623
Precision (macro average)	0.8435
Recall (macro average)	0.8480
F1-score (macro average)	0.8432

The model achieved an overall accuracy of approximately 86%, with strong balance between precision and recall across most categories.

Per-Class Metrics
Class	Precision	Recall	F1-Score	Support
Cardboard	0.9412	0.8533	0.8951	75
Glass	0.9072	0.8544	0.8800	103
Metal	0.7526	0.9359	0.8343	78
Paper	0.8947	0.8947	0.8947	114
Plastic	0.8866	0.8190	0.8515	105
Trash	0.6786	0.7308	0.7037	26

The model performed best on cardboard, glass, and paper, maintaining consistent precision and recall.
However, trash and metal classes showed occasional confusion, likely due to visual similarity in textures or color patterns.

4. Confusion Matrix Analysis

The confusion matrix provides a deeper insight into class-level predictions.
Below is the normalized confusion matrix:

Key Observations

The model consistently identified metal images with a 94% true positive rate, showing strong discriminative capability.

Cardboard and glass were sometimes misclassified as paper, likely due to overlapping visual cues.

Trash showed the highest confusion with plastic and paper, indicating that additional data augmentation or class balancing may help.

5. Inference Simulation

The conveyor belt simulation mimics a real-time waste-sorting scenario. The model processed 100 test images sequentially, logging predictions and confidence values in a CSV file (results/predictions.csv).

Example predictions:

Frame 000: trash (94.90%)
Frame 002: glass (59.13%) LOW CONFIDENCE
Frame 004: paper (99.89%)
Frame 006: plastic (50.50%) LOW CONFIDENCE
Frame 016: cardboard (99.88%)
Frame 045: paper (42.11%) LOW CONFIDENCE
Frame 087: trash (58.56%) LOW CONFIDENCE
Frame 099: paper (97.84%)

Confidence Summary

Most predictions exceeded 90% confidence.

Approximately 8% of samples were flagged as low confidence (below 60% threshold).

Low-confidence predictions mainly appeared between visually similar classes (plastic vs. trash, paper vs. cardboard).

6. Visual Evaluation

A grid visualization of 100 test images was generated to visually confirm model predictions.
Each image was annotated with the predicted class and confidence score.

Example visualization (10x10 grid):

7. Summary and Recommendations
Strengths

The model achieves high accuracy with stable precision and recall across major categories.

Transfer learning from ResNet18 significantly reduces training time while retaining strong feature extraction.

Works effectively even with modest hardware and limited training epochs.

Areas for Improvement

The trash and plastic categories require more diverse training samples to reduce confusion.

Incorporating real-world lighting variations and background diversity can improve generalization.

A lightweight deployment version (e.g., ONNX or TorchScript) has been successfully exported, but further quantization can be explored for edge deployment.

Future Work

Increase training epochs to fine-tune feature extraction further.

Experiment with EfficientNet or Vision Transformer architectures for potential accuracy gains.

Integrate confidence-based filtering to flag uncertain predictions during real-world deployment.

Final Remarks

The TrashNet classification model demonstrates reliable performance for automated waste detection tasks.
With additional data and extended fine-tuning, this system can be a strong foundation for real-time waste-sorting applications in smart recycling systems and sustainability-driven automation.
