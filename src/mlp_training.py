import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# 1. LOAD NUMPY DATA
# ===============================
X = np.load("data/genomes/X_train.npy")        # (n_samples, n_features)
y = np.load("data/genomes/y_train.npy",
            allow_pickle=True)        # (n_samples,)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ===============================
# 2. REMOVE RARE CLASSES (<2)
# ===============================
class_counts = Counter(y)

keep_classes = {cls for cls, cnt in class_counts.items() if cnt >= 2}
mask = np.array([label in keep_classes for label in y])

X = X[mask]
y = y[mask]

print("Filtered X shape:", X.shape)
print("Remaining classes:", len(keep_classes))


# ===============================
# 3. TRAIN / VAL SPLIT (STRATIFIED)
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# 4. SCALE FEATURES
# ===============================
scaler = StandardScaler(with_mean=False)
# with_mean=False is IMPORTANT for sparse / k-mer data

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ===============================
# 5. TRAIN MLP
# ===============================
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20,
    verbose=True
)

mlp.fit(X_train, y_train)


# ===============================
# 6. EVALUATION
# ===============================
y_pred = mlp.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))
