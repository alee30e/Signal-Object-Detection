from typing import Counter

import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn import preprocessing

train_csv = pd.read_csv("../data/raw/train.csv")
test_csv = pd.read_csv("../data/raw/test.csv")
# print(train_csv)
#
# print(train_csv.shape)
# print(train_csv.columns)
# print(train_csv.isnull().any())
#
# duplicates = train_csv["id"].duplicated()
# print(duplicates.sum())
#
# labels = train_csv["label"].unique()
# print(labels)
#
# class_counts = train_csv["label"].value_counts()
# print(class_counts)
#
#
# print(test_csv.head())
# print(test_csv.shape)
# print(test_csv.columns)
#
# print(test_csv.isnull().sum())
# print(test_csv["id"].duplicated().sum())

train_images_dir = Path("../data/raw/train")
cnt = 0
sizes = []
modes = []
arr_img = []
arr_images = []
final_array = []
for id_image in train_csv["id"]:
    image_path = train_images_dir/id_image
    img = Image.open(image_path)
    img_gray = img.convert("L")

    arr_image = np.array(img_gray) / 255.0

    arr_images = arr_image.flatten()

    final_array.append(arr_images)

    sizes.append(img.size)
    modes.append(img.mode)

arr_labels = np.array([int(label) for label in train_csv["label"]])

# plt.imshow(img)
# plt.show()

# test_sizes = []
# test_modes = []
test_images_dir = Path("../data/raw/test")
# arr_img = []
test_array = []
for image_name in test_csv["id"]:
    image_path = test_images_dir / image_name
    img = Image.open(image_path)
    img_gray = img.convert("L")
    array = np.array(img_gray) / 255.0
    test_array.append(array.flatten())

#
# img = Image.open(train_images_dir / train_csv["id"].iloc[0])
# arr = np.array(img)
#
# alpha = arr[:, :, 3]
#
# print(alpha.min())
# print(alpha.max())
#
# img_rgb = img.convert("RGB")
# img_gray = img.convert("L")

# plt.imshow(img_rgb)
# plt.show()
#
# plt.imshow(img_gray, cmap="gray")
# plt.show()

# arr_img = np.array(img_gray)

# print(arr_img.shape)
#
# # arr_img = arr_img/ 255.0
# print(arr_img)
#
# arr_img = arr_img.flatten()
# print(arr_img)
#
# print(arr_labels.shape)



from sklearn.model_selection import train_test_split


scaler = preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(final_array)
X_test_scaled = scaler.transform(test_array)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, arr_labels)

predictions = model.predict(X_test_scaled)

submission = pd.DataFrame({
    "id": test_csv["id"],
    "label": predictions
})

print(submission.head())
print(submission["label"].value_counts().sort_index())

SUBMISSION_PATH = Path("../outputs/submissions") / "submission_sgd.csv"

submission.to_csv(SUBMISSION_PATH, index=False)