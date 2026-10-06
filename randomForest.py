import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


train_csv = pd.read_csv("../data/raw/train.csv")
test_csv = pd.read_csv("../data/raw/test.csv")

train_images_dir = Path("../data/raw/train")
test_images_dir = Path("../data/raw/test")


print(train_csv.shape)
print(train_csv.columns)
print(train_csv.isnull().any())
print(train_csv["id"].duplicated().sum())
print(train_csv["label"].unique())
print(train_csv["label"].value_counts())

print(test_csv.head())
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.isnull().sum())
print(test_csv["id"].duplicated().sum())


def load_image_as_vector(image_path, resize_to=None):
    image = Image.open(image_path)
    image_gray = image.convert("L")

    if resize_to is not None:
        image_gray = image_gray.resize(resize_to)

    image_array = np.array(image_gray, dtype=np.float32)
    image_array = image_array / 255.0

    image_shape = image_array.shape
    image_vector = image_array.flatten()

    return image_vector, image_shape


def load_train(train_csv):
    train_ids = train_csv["id"].to_numpy()
    train_labels = train_csv["label"].astype(int).to_numpy()

    train_images_vector = []
    train_images_shapes = []

    for index, image in enumerate(train_ids):
        path = train_images_dir / image
        image_vector, image_shape = load_image_as_vector(path, resize_to=None)

        train_images_vector.append(image_vector)
        train_images_shapes.append(image_shape)

    train_images = np.array(train_images_vector, dtype=np.float32)

    return train_images, train_labels, train_ids, train_images_shapes


def load_test(test_csv):
    test_ids = test_csv["id"].to_numpy()

    test_images_vector = []
    test_images_shapes = []

    for index, image in enumerate(test_ids):
        path = test_images_dir / image
        image_vector, image_shape = load_image_as_vector(path, resize_to=None)

        test_images_vector.append(image_vector)
        test_images_shapes.append(image_shape)

    test_images = np.array(test_images_vector, dtype=np.float32)

    return test_images, test_ids, test_images_shapes


X_train_images, y_train_images, train_images_ids, train_images_shapes = load_train(train_csv)
X_test_images, test_images_ids, test_images_shape = load_test(test_csv)

print("X_train_images:", X_train_images.shape)
print("y_train_images:", y_train_images.shape)
print("X_test_images:", X_test_images.shape)


X_train_80, X_val_20, y_train_80, y_val_20 = train_test_split(
    X_train_images,
    y_train_images,
    test_size=0.2,
    random_state=42,
    stratify=y_train_images
)


random_forest_configs = [
    {
        "n_estimators": 100,
        "max_depth": 20,
        "max_features": "sqrt"
    },
    {
        "n_estimators": 100,
        "max_depth": 40,
        "max_features": "sqrt"
    },
    {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": "sqrt"
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "max_features": "sqrt"
    },
    {
        "n_estimators": 300,
        "max_depth": 40,
        "max_features": "sqrt"
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "sqrt"
    }
]


best_config = None
best_accuracy = 0
best_model = None

results = []

for config in random_forest_configs:
    print()
    rf_model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train_80, y_train_80)

    val_predictions = rf_model.predict(X_val_20)

    acc = accuracy_score(y_val_20, val_predictions)

    print("Validation accuracy:", acc)

    results.append({
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "max_features": config["max_features"],
        "accuracy": acc
    })

    if acc > best_accuracy:
        best_accuracy = acc
        best_config = config
        best_model = rf_model


print()
print("Rezultate:")
for result in results:
    print(result)

print()
print("Best config:", best_config)
print("Best validation accuracy:", best_accuracy)


best_val_predictions = best_model.predict(X_val_20)

print()
print("Confusion matrix:")
print(confusion_matrix(y_val_20, best_val_predictions))

print()
print("Classification report:")
print(classification_report(y_val_20, best_val_predictions))


final_rf_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    max_features=best_config["max_features"],
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_rf_model.fit(X_train_images, y_train_images)

test_predictions = final_rf_model.predict(X_test_images)


submission = pd.DataFrame({
    "id": test_images_ids,
    "label": test_predictions
})

print()
print(submission.head())

print()
print(submission["label"].value_counts().sort_index())

SUBMISSION_PATH = Path("../outputs/submissions")
SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)

submission_file = SUBMISSION_PATH / "submission_random_forest.csv"

submission.to_csv(submission_file, index=False)

print()
print("Submission salvat la:", submission_file)
# C:\Users\ale_1\PycharmProjects\Signal-Object-Detection\.venv\Scripts\python.exe C:\Users\ale_1\PycharmProjects\Signal-Object-Detection\src\main_randomForest.py
# (15500, 2)
# Index(['id', 'label'], dtype='str')
# id       False
# label    False
# dtype: bool
# 0
# [1 2 5 3 4]
# label
# 1    3500
# 2    3000
# 5    3000
# 3    3000
# 4    3000
# Name: count, dtype: int64
#                                          id
# 0  a5ce7900-4a92-4969-887b-882515678aaa.png
# 1  afac9315-be6d-49de-a02b-56ab6de79285.png
# 2  225ea6a6-6e0a-4c9b-b2ac-8ca32c9fa598.png
# 3  e5fbafa1-58c6-4aea-a3d5-3910483ef377.png
# 4  c3dff3c1-df96-47bc-a41d-be05eadaf701.png
# (5500, 1)
# Index(['id'], dtype='str')
# id    0
# dtype: int64
# 0
# X_train_images: (15500, 7040)
# y_train_images: (15500,)
# X_test_images: (5500, 7040)
#
# Testez Random Forest: {'n_estimators': 100, 'max_depth': 20, 'max_features': 'sqrt'}
# Validation accuracy: 0.26129032258064516
#
# Testez Random Forest: {'n_estimators': 100, 'max_depth': 40, 'max_features': 'sqrt'}
# Validation accuracy: 0.2503225806451613
#
# Testez Random Forest: {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt'}
# Validation accuracy: 0.25483870967741934
#
# Testez Random Forest: {'n_estimators': 300, 'max_depth': 20, 'max_features': 'sqrt'}
# Validation accuracy: 0.2574193548387097
#
# Testez Random Forest: {'n_estimators': 300, 'max_depth': 40, 'max_features': 'sqrt'}
# Validation accuracy: 0.27064516129032257
#
# Testez Random Forest: {'n_estimators': 300, 'max_depth': None, 'max_features': 'sqrt'}
# Validation accuracy: 0.26645161290322583
#
# Rezultate:
# {'n_estimators': 100, 'max_depth': 20, 'max_features': 'sqrt', 'accuracy': 0.26129032258064516}
# {'n_estimators': 100, 'max_depth': 40, 'max_features': 'sqrt', 'accuracy': 0.2503225806451613}
# {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt', 'accuracy': 0.25483870967741934}
# {'n_estimators': 300, 'max_depth': 20, 'max_features': 'sqrt', 'accuracy': 0.2574193548387097}
# {'n_estimators': 300, 'max_depth': 40, 'max_features': 'sqrt', 'accuracy': 0.27064516129032257}
# {'n_estimators': 300, 'max_depth': None, 'max_features': 'sqrt', 'accuracy': 0.26645161290322583}
#
# Best config: {'n_estimators': 300, 'max_depth': 40, 'max_features': 'sqrt'}
# Best validation accuracy: 0.27064516129032257
#
# Confusion matrix:
# [[435  83  76  65  41]
#  [308  90  96  55  51]
#  [262  76  95  79  88]
#  [253  57  98  90 102]
#  [241  58  74  98 129]]
#
# Classification report:
#               precision    recall  f1-score   support
#
#            1       0.29      0.62      0.40       700
#            2       0.25      0.15      0.19       600
#            3       0.22      0.16      0.18       600
#            4       0.23      0.15      0.18       600
#            5       0.31      0.21      0.26       600
#
#     accuracy                           0.27      3100
#    macro avg       0.26      0.26      0.24      3100
# weighted avg       0.26      0.27      0.25      3100
#
#
#                                          id  label
# 0  a5ce7900-4a92-4969-887b-882515678aaa.png      2
# 1  afac9315-be6d-49de-a02b-56ab6de79285.png      1
# 2  225ea6a6-6e0a-4c9b-b2ac-8ca32c9fa598.png      4
# 3  e5fbafa1-58c6-4aea-a3d5-3910483ef377.png      1
# 4  c3dff3c1-df96-47bc-a41d-be05eadaf701.png      4
#
# label
# 1    2628
# 2     668
# 3     734
# 4     708
# 5     762
# Name: count, dtype: int64
#
# Submission salvat la: ..\outputs\submissions\submission_random_forest.csv

