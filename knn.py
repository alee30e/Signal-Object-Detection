import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


train_csv = pd.read_csv("../data/raw/train.csv")
test_csv = pd.read_csv("../data/raw/test.csv")

train_images_dir = Path("../data/raw/train")
test_images_dir = Path("../data/raw/test")


def load_images_as_vectors(csv_file, images_dir):
    final_array = []

    for index, image_name in enumerate(csv_file["id"]):
        image_path = images_dir / image_name

        img = Image.open(image_path).convert("L")

        arr_image = np.array(img).astype("float32") / 255.0
        arr_image = arr_image.flatten()

        final_array.append(arr_image)

        if (index + 1) % 1000 == 0:
            print(f"Loaded {index + 1} images")

    return np.array(final_array, dtype="float32")


print("Loading train images...")
X_train_all = load_images_as_vectors(train_csv, train_images_dir)

print("Loading test images...")
X_test = load_images_as_vectors(test_csv, test_images_dir)

arr_labels = np.array([int(label) for label in train_csv["label"]])

print("X_train_all shape:", X_train_all.shape)
print("X_test shape:", X_test.shape)
print("Labels shape:", arr_labels.shape)

scaler = preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=100, random_state=1)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)

model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    metric="minkowski",
    p=2,
    n_jobs=-1
)

model.fit(X_train_pca, arr_labels)

predictions = model.predict(X_test_pca)

submission = pd.DataFrame({
    "id": test_csv["id"],
    "label": predictions
})

print(submission.head())
print(submission["label"].value_counts().sort_index())

SUBMISSION_PATH = Path("../outputs/submissions") / "submission2knn.csv"
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

submission.to_csv(SUBMISSION_PATH, index=False)

print("Saved submission to:", SUBMISSION_PATH)