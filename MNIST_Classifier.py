from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain

# Load the MNIST dataset
mnist = fetch_openml("mnist_784", as_frame=False, version=1)

X, y = mnist.data, mnist.target
y = y.astype(np.int8)

# Function to display an image
def display_image(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.savefig("MultiOutputClf.png")
    # plt.show()

some_digit = X[0]
# display_image(some_digit)

# Split data into train and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_is_5 = (y_train == 5)
y_test_is_5 = (y_test == 5)

# Train SGDClassifier for binary classification (digit 5)
sgd_clf = SGDClassifier(n_jobs=6, random_state=42)
sgd_clf.fit(X_train, y_train_is_5)
# prediction = sgd_clf.predict([some_digit])
# print(prediction)  # Output: True

# cross_val_scores_sgd = cross_val_score(sgd_clf, X_train, y_train_is_5, cv=3, n_jobs=6)
# print(cross_val_scores_sgd.mean())  # Output: 0.95

# Baseline comparison with DummyClassifier
# dummy_clf = DummyClassifier(random_state=42)

# cross_val_scores_dummy = cross_val_score(dummy_clf, X_train, y_train_is_5, cv=3, n_jobs=6)
# print(cross_val_scores_dummy.mean())  # Output: 0.90

# Generate predictions and compute confusion matrix
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_is_5, cv=3, n_jobs=6)
# conf_matrix = confusion_matrix(y_train_is_5, y_train_pred)
# print(conf_matrix)  # Output: [[53892   687]
#                     #          [ 1891  3530]]

# Compute precision and recall
# precision_sgd = precision_score(y_train_is_5, y_train_pred)
# print(precision_sgd)
# recall_sgd = recall_score(y_train_is_5, y_train_pred)
# print(recall_sgd)

# y_score = sgd_clf.decision_function([some_digit])
# print(y_score)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_is_5, cv=3, method="decision_function")

precisions_sgd, recalls_sgd, thresholds_sgd = precision_recall_curve(y_train_is_5, y_scores)

# Plot precision and recall curves
# plt.figure()
# plt.plot(thresholds_sgd, precisions_sgd[:-1], "g-", label="Precision", linewidth=2)
# plt.plot(thresholds_sgd, recalls_sgd[:-1], "r-", label="Recall", linewidth=2)
# Plot precision-recall curve
# plt.plot(recalls_sgd, precisions_sgd, label="Precision-Recall Curve", linewidth=3)
# plt.legend()
# plt.grid()
# plt.yticks(np.arange(0, 1.1, 0.05))
# plt.show()

idx_for_90_precision_sgd = (precisions_sgd >= 0.9).argmax()
threshold_for_90_precision_sgd = thresholds_sgd[idx_for_90_precision_sgd]

# plt.figure()
# plt.plot(thresholds_sgd, precisions_sgd[:-1], "g-", label="Precision", linewidth=2)
# plt.plot(thresholds_sgd, recalls_sgd[:-1], "r-", label="Recall", linewidth=2)
# plt.vlines(threshold_for_90_precision_sgd, 0, 1, "k", "dotted", label="Threshold at 90% Precision", linewidth=3)
# plt.legend()
# plt.grid()
# plt.yticks(np.arange(0, 1.1, 0.05))
# plt.show()

# y_train_pred_90_sgd = (y_scores >= threshold_for_90_precision_sgd)
# precision_value = precision_score(y_train_is_5, y_train_pred_90_sgd)
# print(precision_value)
# recall_value = recall_score(y_train_is_5, y_train_pred_90_sgd)
# print(recall_value)

# Compute ROC curve
fpr_sgd, tpr_sgd, thresholds_sgd_roc = roc_curve(y_train_is_5, y_scores)

idx_for_90_precision_sgd_roc = (thresholds_sgd_roc <= threshold_for_90_precision_sgd).argmax()
fpr_sgd_at_90, tpr_sgd_at_90 = fpr_sgd[idx_for_90_precision_sgd_roc], tpr_sgd[idx_for_90_precision_sgd_roc]

# Plot ROC curve
# plt.figure()
# plt.plot(fpr_sgd, tpr_sgd, "b-", label="ROC Curve", linewidth=1)
# plt.plot(fpr_sgd_at_90, tpr_sgd_at_90, "ro", label="Point at 90% Precision")
# plt.legend()
# plt.grid()
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate (Recall)")
# plt.show()

# roc_auc_sgd = roc_auc_score(y_train_is_5, y_scores)
# print(fpr_sgd_at_90)
# print(roc_auc_sgd)

# Train RandomForestClassifier for binary classification
rnd_clf = RandomForestClassifier(random_state=42, n_jobs=6)
rnd_clf.fit(X_train, y_train_is_5)
# prediction = rnd_clf.predict([some_digit])
# print(prediction)  # Output: True

# cross_val_scores_rnd = cross_val_score(rnd_clf, X_train, y_train_is_5, n_jobs=6)
# print(cross_val_scores_rnd.mean())  # Output: 0.98

y_probas_forest = cross_val_predict(rnd_clf, X_train, y_train_is_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
# print(y_probas_forest[:10])

precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_is_5, y_scores_forest)

idx_for_90_precision_forest = (precisions_forest >= 0.9).argmax()
threshold_for_90_precision_forest = thresholds_forest[idx_for_90_precision_forest]

y_train_pred_90_forest = (y_scores_forest >= threshold_for_90_precision_forest)

# plt.figure()
# plt.plot(thresholds_forest, precisions_forest[:-1], "g-", label="Precision", linewidth=2)
# plt.plot(thresholds_forest, recalls_forest[:-1], "r--", label="Recall", linewidth=2)
# plt.vlines(threshold_for_90_precision_forest, 0, 1, "k", "dotted", label="Threshold at 90% Precision")
# plt.plot(recalls_forest, precisions_forest, label="Precision-Recall Curve")
# plt.legend()
# plt.grid()
# plt.yticks(np.arange(0, 1.1, 0.05))
# plt.show()

# precision_value = precision_score(y_train_is_5, y_train_pred_90_forest)
# print(precision_value)  # Output: 0.90
# recall_value = recall_score(y_train_is_5, y_train_pred_90_forest)
# print(recall_value)  # Output: 0.97

fpr_forest, tpr_forest, thresholds_forest_roc = roc_curve(y_train_is_5, y_scores_forest)
idx_for_90_precision_forest_roc = (thresholds_forest_roc <= threshold_for_90_precision_forest).argmax()
fpr_forest_at_90, tpr_forest_at_90 = fpr_forest[idx_for_90_precision_forest_roc], tpr_forest[idx_for_90_precision_forest_roc]

# plt.figure()
# plt.plot(fpr_forest, tpr_forest, "g-", label="ROC Curve", linewidth=2)
# plt.plot(fpr_forest_at_90, tpr_forest_at_90, "ro", label="Point at 90% Precision")
# plt.legend()
# plt.grid()
# plt.yticks(np.arange(0, 1.1, 0.05))
# plt.show()

# roc_auc_forest = roc_auc_score(y_train_is_5, y_scores_forest)
# print(fpr_forest_at_90)  # Output: 0.01
# print(roc_auc_forest)  # Output: 0.998

# Multiclass Classification

# svm_clf = SVC(random_state=42)
# svm_clf.decision_function_shape = "ovo"
# svm_clf.fit(X_train[:2000], y_train[:2000])

# some_digit_scores = svm_clf.decision_function([some_digit])
# print(some_digit_scores)  # Output: array of scores
# print(svm_clf.predict([some_digit]))  # Output: 5
# print(svm_clf.classes_)  # Output: array of classes
# class_id = some_digit_scores.argmax()
# print(class_id)  # Output: 5

# ovr_clf = OneVsRestClassifier(SVC(random_state=42), n_jobs=6)
# ovr_clf.fit(X_train, y_train)

# some_digit_scores = ovr_clf.decision_function([some_digit])
# print(some_digit_scores)

sgd_clf_multiclass = SGDClassifier(random_state=42, n_jobs=6)
# sgd_clf_multiclass.fit(X_train, y_train)

# some_digit_scores = sgd_clf_multiclass.decision_function([some_digit])
# print(some_digit_scores)  # Output: array of scores (OvR strategy)
# print(sgd_clf_multiclass.predict([some_digit]))  # Output: 3 (incorrect prediction)

# cross_val_scores = cross_val_score(sgd_clf_multiclass, X_train, y_train, cv=3, n_jobs=6)
# print(cross_val_scores.mean())  # Output: 0.86

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# cross_val_scores = cross_val_score(sgd_clf_multiclass, X_train_scaled, y_train, cv=3, n_jobs=6)
# print(cross_val_scores.mean())  # Output: 0.90

y_train_pred = cross_val_predict(sgd_clf_multiclass, X_train_scaled, y_train, cv=3, n_jobs=6)
sample_weights = (y_train_pred != y_train)

# ConfusionMatrixDisplay.from_predictions(y_true=y_train, y_pred=y_train_pred, normalize="true", values_format=".0%", sample_weight=sample_weights)
# plt.show()

# Visualize errors between two classes
# class_a, class_b = 8, 0

# X_aa = X_train[(y_train == class_a) & (y_train_pred == class_a)]
# X_ab = X_train[(y_train == class_a) & (y_train_pred == class_b)]
# X_bb = X_train[(y_train == class_b) & (y_train_pred == class_b)]
# X_ba = X_train[(y_train == class_b) & (y_train_pred == class_a)]

# size = 5
# pad = 0.2
# plt.figure(figsize=(size, size))
# for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
#                                        (X_aa, (0, 1)), (X_ab, (1, 1))]:
#     for idx, image_data in enumerate(images[:size*size]):
#         x = idx % size + label_col * (size + pad)
#         y = idx // size + label_row * (size + pad)
#         plt.imshow(image_data.reshape(28, 28), cmap="binary",
#                    extent=(x, x + 1, y, y + 1))
# plt.xticks([size / 2, size + pad + size / 2], [str(class_a), str(class_b)])
# plt.yticks([size / 2, size + pad + size / 2], [str(class_b), str(class_a)])
# plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
# plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
# plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
# plt.xlabel("Predicted label")
# plt.ylabel("True label")

# plt.savefig("8 and 0.png")
# plt.show()

# Multilabel Classification

y_train_large = (y_train >= 7)
y_train_odd = (y_train.astype("int8") % 2 == 1)
y_train_prime = [digit in [2, 3, 5, 7] for digit in y_train.astype("int8")]

y_multilabel = np.c_[y_train_large, y_train_odd, y_train_prime]

knn_clf = KNeighborsClassifier(n_jobs=-1)
# knn_clf.fit(X_train[:2000], y_multilabel[:2000])

# print(knn_clf.predict([some_digit]))  # Output: [[False  True  True]]
# print(knn_clf.predict_proba([some_digit]))  # Output: [array([[1., 0.]]), array([[0., 1.]]), array([[0., 1.]])]

# cross_val_scores_knn = cross_val_score(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1, scoring="f1_macro", verbose=2)
# print(cross_val_scores_knn.mean())  # Output: 0.977

chain_clf = ClassifierChain(SVC(decision_function_shape="ovr"), random_state=42, cv=3)

# chain_clf.fit(X_train[:2000], y_multilabel[:2000])

# print(chain_clf.predict([some_digit]))

# cross_val_scores_chain = cross_val_score(chain_clf, X_train[:10000], y_multilabel[:10000], scoring="f1_macro", n_jobs=-1, cv=3, verbose=2)

# print(cross_val_scores_chain.mean())  # Output: 0.96

# Multioutput Classification

np.random.seed(42)
noise_train = np.random.randint(0, 100, (len(X_train), 784))  # 60000 rows, 784 columns
noise_test = np.random.randint(0, 100, (len(X_test), 784))  # 10000 rows, 784 columns
X_train_noisy = X_train + noise_train
X_test_noisy = X_test + noise_test
y_train_clean = X_train

# print(X_train_noisy[0])
noisy_digit = X_train_noisy[0]
display_image(noisy_digit)

knn_clf.fit(X_train_noisy, y_train_clean)

clean_digit = knn_clf.predict([X_test_noisy[0]])

# display_image(clean_digit)  # Display cleaned image