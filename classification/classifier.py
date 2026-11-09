"""Portfolio adaptation of the original Lab 2 console classification workflow.

Runs the five validation-selected configurations from the final report, not
the full console hyperparameter sweep. Metrics may differ from the report
because image ordering, library versions, and random states affect results.
Requires authorized local PNG images in 'Utensil images/{FORK,KNIFE,SPOON}'.
"""
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ("FORK", "KNIFE", "SPOON")
DATA = Path("Utensil images")


def read_images():
    features, labels = [], []
    for label, cls in enumerate(CLASSES):
        paths = sorted((DATA / cls).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG images in {DATA / cls}")
        for path in paths:
            img = imread(path)
            if img.ndim == 3 and img.shape[2] == 4:
                img = rgba2rgb(img)
            if img.ndim == 3:
                img = rgb2gray(img)
            img = resize(img, (64, 64), anti_aliasing=True)
            features.append(hog(img, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2)))
            labels.append(label)
    return np.asarray(features), np.asarray(labels)


def main():
    X, y = read_images()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    models = {
        "Logistic Regression": LogisticRegression(
            penalty=None, solver="newton-cg", max_iter=300,
            class_weight="balanced"),
        "KNN": KNeighborsClassifier(
            n_neighbors=3, weights="uniform", p=2, metric="minkowski",
            leaf_size=30),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", splitter="random", max_depth=22,
            min_samples_split=6, min_samples_leaf=3, max_features="log2",
            max_leaf_nodes=60, min_impurity_decrease=0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=90, criterion="entropy", max_depth=12,
            min_samples_split=4, min_samples_leaf=1, max_features="log2",
            bootstrap=False, random_state=42),
        "SVC": SVC(kernel="rbf", C=2.5, tol=0.0005, max_iter=4000,
                   gamma=0.07),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        X_full = np.concatenate([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        model.fit(X_full, y_full)
        predicted = model.predict(X_test)
        print(f"\\n{name} | validation: {val_accuracy:.2%} | "
              f"test: {accuracy_score(y_test, predicted):.2%}")
        print(confusion_matrix(y_test, predicted, labels=[0, 1, 2]))
        print(classification_report(y_test, predicted,
                                    labels=[0, 1, 2],
                                    target_names=CLASSES,
                                    zero_division=0))
    print("Portfolio run, not an exact reproduction of historical console outputs.")


if __name__ == "__main__":
    main()
