import matplotlib
matplotlib.use("Agg")

import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


train_csv = pd.read_csv("../data/raw/train.csv")
test_csv = pd.read_csv("../data/raw/test.csv")

train_images_dir = Path("../data/raw/train")
test_images_dir = Path("../data/raw/test")

SUBMISSION_PATH = Path("../outputs/submissions")
SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)

PLOTS_PATH = Path("../outputs/plots/projections_statistics")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

STATS_PATH = Path("../outputs/statistics")
STATS_PATH.mkdir(parents=True, exist_ok=True)


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


def load_image_array(image_path, resize_to=None):
    image = Image.open(image_path)
    image_gray = image.convert("L")

    if resize_to is not None:
        image_gray = image_gray.resize(resize_to)

    image_array = np.array(image_gray, dtype=np.float32)
    image_array = image_array / 255.0

    return image_array


def extract_projection_statistics_features(image_array):
    raw_pixels = image_array.flatten()

    horizontal_projection = image_array.sum(axis=1)
    vertical_projection = image_array.sum(axis=0)

    mean_value = image_array.mean()
    std_value = image_array.std()
    min_value = image_array.min()
    max_value = image_array.max()
    sum_value = image_array.sum()
    energy = np.sum(image_array ** 2)

    pixels_above_025 = np.sum(image_array > 0.25)
    pixels_above_050 = np.sum(image_array > 0.50)
    pixels_above_075 = np.sum(image_array > 0.75)

    ratio_above_025 = np.mean(image_array > 0.25)
    ratio_above_050 = np.mean(image_array > 0.50)
    ratio_above_075 = np.mean(image_array > 0.75)

    horizontal_mean = horizontal_projection.mean()
    horizontal_std = horizontal_projection.std()
    horizontal_max = horizontal_projection.max()

    vertical_mean = vertical_projection.mean()
    vertical_std = vertical_projection.std()
    vertical_max = vertical_projection.max()

    statistics = np.array([
        mean_value,
        std_value,
        min_value,
        max_value,
        sum_value,
        energy,
        pixels_above_025,
        pixels_above_050,
        pixels_above_075,
        ratio_above_025,
        ratio_above_050,
        ratio_above_075,
        horizontal_mean,
        horizontal_std,
        horizontal_max,
        vertical_mean,
        vertical_std,
        vertical_max
    ], dtype=np.float32)

    image_features = np.concatenate([
        raw_pixels,
        horizontal_projection,
        vertical_projection,
        statistics
    ]).astype(np.float32)

    stats_dict = {
        "mean_value": mean_value,
        "std_value": std_value,
        "min_value": min_value,
        "max_value": max_value,
        "sum_value": sum_value,
        "energy": energy,
        "pixels_above_025": pixels_above_025,
        "pixels_above_050": pixels_above_050,
        "pixels_above_075": pixels_above_075,
        "ratio_above_025": ratio_above_025,
        "ratio_above_050": ratio_above_050,
        "ratio_above_075": ratio_above_075,
        "horizontal_mean": horizontal_mean,
        "horizontal_std": horizontal_std,
        "horizontal_max": horizontal_max,
        "vertical_mean": vertical_mean,
        "vertical_std": vertical_std,
        "vertical_max": vertical_max
    }

    return image_features, stats_dict, horizontal_projection, vertical_projection


def load_image_as_features(image_path, resize_to=None):
    image_array = load_image_array(image_path, resize_to=resize_to)

    image_features, stats_dict, horizontal_projection, vertical_projection = extract_projection_statistics_features(
        image_array
    )

    image_shape = image_array.shape

    return image_features, image_shape, stats_dict


def load_train(train_csv):
    train_ids = train_csv["id"].to_numpy()
    train_labels = train_csv["label"].astype(int).to_numpy()

    train_images_vector = []
    train_images_shapes = []
    train_stats_rows = []

    for index, image in enumerate(train_ids):
        path = train_images_dir / image
        image_vector, image_shape, stats_dict = load_image_as_features(path, resize_to=None)

        train_images_vector.append(image_vector)
        train_images_shapes.append(image_shape)

        stats_dict["id"] = image
        stats_dict["label"] = train_labels[index]
        train_stats_rows.append(stats_dict)

        if (index + 1) % 1000 == 0:
            print("Am incarcat train:", index + 1)

    train_images = np.array(train_images_vector, dtype=np.float32)
    train_stats = pd.DataFrame(train_stats_rows)

    return train_images, train_labels, train_ids, train_images_shapes, train_stats


def load_test(test_csv):
    test_ids = test_csv["id"].to_numpy()

    test_images_vector = []
    test_images_shapes = []
    test_stats_rows = []

    for index, image in enumerate(test_ids):
        path = test_images_dir / image
        image_vector, image_shape, stats_dict = load_image_as_features(path, resize_to=None)

        test_images_vector.append(image_vector)
        test_images_shapes.append(image_shape)

        stats_dict["id"] = image
        test_stats_rows.append(stats_dict)

        if (index + 1) % 1000 == 0:
            print("Am incarcat test:", index + 1)

    test_images = np.array(test_images_vector, dtype=np.float32)
    test_stats = pd.DataFrame(test_stats_rows)

    return test_images, test_ids, test_images_shapes, test_stats


def save_class_distribution_plot(train_csv):
    counts = train_csv["label"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Clasa")
    plt.ylabel("Numar imagini")
    plt.title("Distributia claselor in train")
    plt.tight_layout()

    output_file = PLOTS_PATH / "class_distribution.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Plot salvat:", output_file)


def save_projection_examples_plot(train_csv):
    labels = sorted(train_csv["label"].unique())

    fig, axes = plt.subplots(len(labels), 3, figsize=(13, 3 * len(labels)))

    for row_index, label in enumerate(labels):
        image_id = train_csv[train_csv["label"] == label]["id"].iloc[0]
        image_path = train_images_dir / image_id

        image_array = load_image_array(image_path, resize_to=None)
        image_features, stats_dict, horizontal_projection, vertical_projection = extract_projection_statistics_features(
            image_array
        )

        axes[row_index, 0].imshow(image_array, cmap="gray")
        axes[row_index, 0].set_title(f"Imagine originala - clasa {label}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].plot(horizontal_projection)
        axes[row_index, 1].set_title(f"Proiectie orizontala - clasa {label}")
        axes[row_index, 1].set_xlabel("Rand")
        axes[row_index, 1].set_ylabel("Suma intensitati")

        axes[row_index, 2].plot(vertical_projection)
        axes[row_index, 2].set_title(f"Proiectie verticala - clasa {label}")
        axes[row_index, 2].set_xlabel("Coloana")
        axes[row_index, 2].set_ylabel("Suma intensitati")

    plt.tight_layout()

    output_file = PLOTS_PATH / "projection_examples_per_class.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Plot salvat:", output_file)


def save_statistics_boxplots(train_stats):
    stats_columns = [
        "mean_value",
        "std_value",
        "sum_value",
        "energy",
        "ratio_above_025",
        "ratio_above_050",
        "ratio_above_075",
        "horizontal_std",
        "vertical_std"
    ]

    labels = sorted(train_stats["label"].unique())

    for column in stats_columns:
        data = [
            train_stats[train_stats["label"] == label][column].values
            for label in labels
        ]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=[str(label) for label in labels])
        plt.xlabel("Clasa")
        plt.ylabel(column)
        plt.title(f"Distributia statisticii {column} pe clase")
        plt.tight_layout()

        output_file = PLOTS_PATH / f"boxplot_{column}.png"
        plt.savefig(output_file, dpi=200)
        plt.close()

        print("Plot salvat:", output_file)


def save_statistics_mean_by_class(train_stats):
    stats_columns = [
        "mean_value",
        "std_value",
        "sum_value",
        "energy",
        "ratio_above_025",
        "ratio_above_050",
        "ratio_above_075",
        "horizontal_std",
        "vertical_std"
    ]

    grouped = train_stats.groupby("label")[stats_columns].mean()
    grouped.to_csv(STATS_PATH / "train_statistics_mean_by_class.csv")

    for column in stats_columns:
        plt.figure(figsize=(7, 5))
        plt.bar(grouped.index.astype(str), grouped[column].values)
        plt.xlabel("Clasa")
        plt.ylabel(column)
        plt.title(f"Media statisticii {column} pe clase")
        plt.tight_layout()

        output_file = PLOTS_PATH / f"mean_by_class_{column}.png"
        plt.savefig(output_file, dpi=200)
        plt.close()

        print("Plot salvat:", output_file)


def save_accuracy_plot(results):
    config_names = []

    for result in results:
        name = (
            "n="
            + str(result["n_estimators"])
            + ", depth="
            + str(result["max_depth"])
        )
        config_names.append(name)

    accuracies = [result["accuracy"] for result in results]

    plt.figure(figsize=(10, 5))
    plt.plot(config_names, accuracies, marker="o")
    plt.xlabel("Configuratie")
    plt.ylabel("Validation accuracy")
    plt.title("Extra Trees - raw pixels + proiectii + statistici")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True)
    plt.tight_layout()

    output_file = PLOTS_PATH / "extra_trees_accuracy_by_config.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Plot salvat:", output_file)


def save_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(y_true))

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion matrix - Extra Trees + proiectii + statistici")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    output_file = PLOTS_PATH / "extra_trees_confusion_matrix.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Plot salvat:", output_file)


def save_prediction_distribution_plot(submission):
    counts = submission["label"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Clasa prezisa")
    plt.ylabel("Numar imagini")
    plt.title("Distributia predictiilor pe test")
    plt.tight_layout()

    output_file = PLOTS_PATH / "test_prediction_distribution.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Plot salvat:", output_file)


save_class_distribution_plot(train_csv)
save_projection_examples_plot(train_csv)


X_train_images, y_train_images, train_images_ids, train_images_shapes, train_stats = load_train(train_csv)
X_test_images, test_images_ids, test_images_shape, test_stats = load_test(test_csv)

train_stats.to_csv(STATS_PATH / "train_image_statistics.csv", index=False)
test_stats.to_csv(STATS_PATH / "test_image_statistics.csv", index=False)

save_statistics_boxplots(train_stats)
save_statistics_mean_by_class(train_stats)

print("X_train_images:", X_train_images.shape)
print("y_train_images:", y_train_images.shape)
print("X_test_images:", X_test_images.shape)

print("Train stats saved:", STATS_PATH / "train_image_statistics.csv")
print("Test stats saved:", STATS_PATH / "test_image_statistics.csv")


X_train_80, X_val_20, y_train_80, y_val_20 = train_test_split(
    X_train_images,
    y_train_images,
    test_size=0.2,
    random_state=42,
    stratify=y_train_images
)


extra_trees_configs = [
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

for config in extra_trees_configs:
    print()
    print("Testez Extra Trees:", config)

    et_model = ExtraTreesClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    et_model.fit(X_train_80, y_train_80)

    val_predictions = et_model.predict(X_val_20)

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
        best_model = et_model


results_df = pd.DataFrame(results)
results_df.to_csv(STATS_PATH / "extra_trees_projection_statistics_results.csv", index=False)

print()
print("Rezultate:")
for result in results:
    print(result)

print()
print("Best config:", best_config)
print("Best validation accuracy:", best_accuracy)


save_accuracy_plot(results)


best_val_predictions = best_model.predict(X_val_20)

print()
print("Confusion matrix:")
print(confusion_matrix(y_val_20, best_val_predictions))

print()
print("Classification report:")
print(classification_report(y_val_20, best_val_predictions))


save_confusion_matrix_plot(y_val_20, best_val_predictions)


final_et_model = ExtraTreesClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    max_features=best_config["max_features"],
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_et_model.fit(X_train_images, y_train_images)

test_predictions = final_et_model.predict(X_test_images)


submission = pd.DataFrame({
    "id": test_images_ids,
    "label": test_predictions
})

print()
print(submission.head())

print()
print(submission["label"].value_counts().sort_index())


save_prediction_distribution_plot(submission)


submission_file = SUBMISSION_PATH / "submission_proj_stats.csv"

submission.to_csv(submission_file, index=False)

print()
print("Submission salvat la:", submission_file)