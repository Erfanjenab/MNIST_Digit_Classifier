## MNIST_Digit_Classifier

# Overview
This repository implements various classification strategies on the MNIST handwritten digit dataset (70,000 images of 28x28 pixels). We use Scikit-Learn models like SGDClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, and more to tackle:
- Binary Classification: Is it a 5 or not? ⚖️
- Multiclass Classification: Identify digits 0-9. 🔢
- Multilabel Classification: Labels like "large" (≥7), "odd", and "prime". 🏷️
- Multioutput Classification: Denoise noisy images pixel-by-pixel. 🧹
The code is modular, commented, and ready for experimentation. Key highlights include cross-validation, precision-recall curves, ROC analysis, and error visualization.
# Features
📊 Data Loading & Preprocessing: Fetches MNIST via fetch_openml, splits into train/test (60k/10k), and scales features for better performance.
🤖 Models Implemented:
SGDClassifier (binary & multiclass)
RandomForestClassifier (binary)
SVC (multiclass with OvO/OvR)
KNeighborsClassifier (multilabel & multioutput)
ClassifierChain (multilabel chaining)
📉 Evaluation Metrics: Accuracy, Precision, Recall, F1, Confusion Matrices, ROC AUC, Precision-Recall Curves.
🖼️ Visualizations: Image displays, confusion matrix heatmaps, error grids (e.g., misclassifications between 8 and 0).
🔍 Advanced Techniques: Threshold tuning for 90% precision, noise addition for multioutput denoising.

# Usage
Run the main script:
python mnist_classifier.py
Key Sections:
Uncomment lines to train models, compute metrics, or generate plots (e.g., plt.show() or print() statements).
For binary classification: Focus on SGD/RandomForest sections.
For denoising: Check the multioutput section—adds random noise (0-100) and uses KNN to clean pixels.
Customize: Change random_state=42 for reproducibility, or adjust n_jobs for parallel processing.
Example output: Precision at 90% for SGD binary classifier ~0.90 with recall ~0.82 (tune thresholds as needed).
# Results & Metrics 📊
Here's a breakdown of performance metrics from cross-validation (CV=3 unless specified). All on train set (60k samples). 🎯
Binary Classification (Detecting '5') ⚖️
SGDClassifier:
Accuracy: ~95% (cross-val mean)
Confusion Matrix Example: [[53892, 687], [1891, 3530]] (TN, FP / FN, TP)
Precision/Recall (default): ~84% / ~65%
At 90% Precision: Recall ~82%, Threshold ~3000 (from decision_function)
ROC AUC: ~0.96
FPR at 90% Precision: ~0.005
RandomForestClassifier:
Accuracy: ~98% (cross-val mean)
Precision/Recall at 90% Precision: 90% / ~97%
ROC AUC: ~0.998
FPR at 90% Precision: ~0.01
Baseline (DummyClassifier): Accuracy ~90% (always predicts non-5)
Multiclass Classification (Digits 0-9) 🔢
SGDClassifier (OvR strategy):
Accuracy (raw): ~86%
Accuracy (scaled features): ~90% (using StandardScaler)
Error Analysis: High confusion between similar digits (e.g., 8 misclassified as 0—see visualizations).
SVC (OvO/OvR): Trained on subset (2k samples) for demo; scales to full with n_jobs.
# Multilabel Classification (Large, Odd, Prime) 🏷️
Labels: Large (≥7), Odd (%2==1), Prime (2,3,5,7)
KNeighborsClassifier:
F1 Macro Score: ~97.7% (CV=3 on full data)
ClassifierChain (SVC base):
F1 Macro Score: ~96% (CV=3 on 10k subset)
# Multioutput Classification (Image Denoising) 🧹
Adds noise (0-100 per pixel) to images.
KNeighborsClassifier: Predicts each of 784 pixels as a separate label.
Output: Cleaned image from noisy input (visualized in MultiOutputClf.png).
Effective on test set; no quantitative metric but visually accurate.
Overall Insights: RandomForest excels in binary tasks. Scaling boosts multiclass by 4%. Multilabel achieves near-perfect F1 with KNN. 🚀
# Visualizations 🖼️
Precision-Recall Curves: For SGD & RandomForest (uncomment plots).
ROC Curves: Blue for SGD, Green for Forest.
Error Grid: Misclassifications (e.g., 8 vs 0) in 8 and 0.png.
Denoising Example: Noisy vs Clean in MultiOutputClf.png.
