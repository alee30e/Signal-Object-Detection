import copy, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


seed = 42
img_h, img_w = 192, 84
batch_size = 48
initial_epochs, fine_tune_epochs = 85, 30
initial_lr, fine_lr = 3e-4, 2.5e-5
weight_decay_initial, weight_decay_fine = 4e-4, 2.5e-4
warmup_epochs = 5
reg_weight = 0.10
expected_weight = 0.06


data_dir = Path("../data/raw")
train_csv_path = data_dir / "train.csv"
test_csv_path = data_dir / "test.csv"
train_images_dir = data_dir / "train"
test_images_dir = data_dir / "test"
output_dir = Path("../outputs")
submissions_dir = output_dir / "submissions"
models_dir = output_dir / "models"
stats_dir = output_dir / "statistics"
figures_dir = output_dir / "figures"

for folder in [submissions_dir, models_dir, stats_dir, figures_dir]:
    folder.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "best_custom_cnn_final.pth"
ema_path = models_dir / "best_custom_cnn_final_ema.pth"
history_path = stats_dir / "training_history_final.csv"
submission_path = submissions_dir / "submission_cnn_ema_tta_final.csv"
confusion_matrix_path = figures_dir / "confusion_matrix_final.png"
comparison_plot_path = figures_dir / "model_comparison_final.png"


def fix_seed(value=42):
    # fixez seed-ul pentru rezultate mai stabile
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
fix_seed(seed)

train_df = pd.read_csv(train_csv_path, dtype={"id": str})
test_df = pd.read_csv(test_csv_path, dtype={"id": str})

train_df.columns = [str(c).strip().lower() for c in train_df.columns]
test_df.columns = [str(c).strip().lower() for c in test_df.columns]
train_df["id"] = train_df["id"].astype(str).str.strip()
test_df["id"] = test_df["id"].astype(str).str.strip()

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("class counts:")
print(train_df["label"].value_counts().sort_index())

train_df["target"] = train_df["label"].astype(int) - 1


train_part_df, val_df = train_test_split(
    train_df,
    test_size=0.16,
    random_state=seed,
    stratify=train_df["target"]
)

train_part_df = train_part_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

num_classes = train_df["target"].nunique()
print(f"train: {train_part_df.shape} | val: {val_df.shape} | test: {test_df.shape}")

class_weights = torch.tensor(
    compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_part_df["target"].values
    ),
    dtype=torch.float32
)

train_transform = T.Compose([
    T.Resize((img_h, img_w)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=3, translate=(0.035, 0.035), scale=(0.97, 1.04), shear=1.5),
    T.ColorJitter(brightness=0.10, contrast=0.16, saturation=0.06, hue=0.008),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
    T.RandomErasing(p=0.06, scale=(0.002, 0.015), ratio=(0.5, 2.0))
])

fine_transform = T.Compose([
    T.Resize((img_h, img_w)),
    T.RandomHorizontalFlip(p=0.25),
    T.RandomAffine(degrees=2, translate=(0.015, 0.015), scale=(0.985, 1.015), shear=0.8),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3)
])

eval_transform = T.Compose([
    T.Resize((img_h, img_w)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3)
])

def get_png_path(images_dir, image_id):
    image_id = str(image_id)
    if image_id.endswith(".png"):
        return Path(images_dir) / image_id
    return Path(images_dir) / f"{image_id}.png"


class SignalDataset(Dataset):
    def __init__(self, df, images_dir, transform, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.has_labels = has_labels
        self.ids = self.df["id"].astype(str).tolist()
        self.paths = [get_png_path(self.images_dir, img_id) for img_id in self.ids]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)
        if self.has_labels:
            label = int(self.df.iloc[index]["target"])
            return image, torch.tensor(label, dtype=torch.long)
        return image, self.ids[index]


def make_loader(dataset, shuffle_data):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)

train_loader = make_loader(SignalDataset(train_part_df, train_images_dir, train_transform), True)
train_fine_loader = make_loader(SignalDataset(train_part_df, train_images_dir, fine_transform), True)
val_loader = make_loader(SignalDataset(val_df, train_images_dir, eval_transform), False)
test_loader = make_loader(SignalDataset(test_df, test_images_dir, eval_transform, has_labels=False), False)
print(f"batches: train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

class DropPath(nn.Module):
    # opreste uneori o ramura reziduala ca regularizare
    def __init__(self, prob=0.0):
        super().__init__()
        self.prob = prob
    def forward(self, x):
        if not self.training or self.prob == 0:
            return x
        keep = 1 - self.prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape) < keep).float() / keep
        return x * mask

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.shape[:2]
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SEBlock(out_channels)
        self.drop_path = DropPath(drop_prob)
        self.activation = nn.SiLU(inplace=True)
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.drop_path(out)
        return self.activation(out + self.shortcut(x))

class SignalCNN(nn.Module):
    def __init__(self, num_classes, drop_rate=0.06):
        super().__init__()
        drops = [drop_rate * i / 6 for i in range(7)]
        # Blocul initial extrage trasaturi simple.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, 3, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            ConvBlock(48, 64, 2, drops[0]),
            ConvBlock(64, 128, 2, drops[1]),
            ConvBlock(128, 256, 2, drops[2]),
            ConvBlock(256, 384, 2, drops[3]),
            ConvBlock(384, 512, 1, drops[4]),
            ConvBlock(512, 512, 1, drops[5]),
            ConvBlock(512, 512, 1, drops[6])
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Dropout(0.32),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18)
        )

        self.class_head = nn.Linear(256, num_classes)
        self.count_head = nn.Linear(256, 1)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, return_count=False):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        logits = self.class_head(x)
        count_pred = self.count_head(x).squeeze(1)
        if return_count:
            return logits, count_pred
        return logits

@torch.no_grad()
def update_ema(model, ema_model, decay=0.995):
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    for key in ema_state:
        if ema_state[key].dtype.is_floating_point:
            ema_state[key].mul_(decay).add_(model_state[key], alpha=1 - decay)
        else:
            ema_state[key].copy_(model_state[key])

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer)
    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            min_factor = self.eta_min / self.base_lrs[0]
            factor = min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]

class_positions = torch.arange(num_classes).float()

def total_loss(logits, count_pred, labels, criterion):
    ce_loss = criterion(logits, labels)
    count_loss = F.smooth_l1_loss(count_pred, labels.float())
    probs = torch.softmax(logits, dim=1)
    expected_class = (probs * class_positions).sum(dim=1)
    expected_class_loss = F.smooth_l1_loss(expected_class, labels.float())
    return ce_loss + reg_weight * count_loss + expected_weight * expected_class_loss

def train_epoch(model, ema_model, loader, criterion, optimizer, ema_decay):
    model.train()
    loss_sum = 0.0
    preds_all, labels_all = [], []
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, count_pred = model(images, return_count=True)
        loss = total_loss(logits, count_pred, labels, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        update_ema(model, ema_model, ema_decay)
        loss_sum += loss.item() * images.size(0)
        preds_all.extend(logits.argmax(1).detach().numpy())
        labels_all.extend(labels.detach().numpy())
    return loss_sum / len(loader.dataset), accuracy_score(labels_all, preds_all)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    preds_all, labels_all = [], []

    for images, labels in loader:
        logits, count_pred = model(images, return_count=True)
        loss = total_loss(logits, count_pred, labels, criterion)
        loss_sum += loss.item() * images.size(0)
        preds_all.extend(logits.argmax(1).detach().numpy())
        labels_all.extend(labels.detach().numpy())
    return loss_sum / len(loader.dataset), accuracy_score(labels_all, preds_all), np.array(labels_all), np.array(preds_all)

def run_training_phase(phase_name, model, ema_model, loader, criterion, optimizer,
                       scheduler, epochs, patience, ema_decay,
                       best_val, best_ema, epoch_offset, history):
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        lr = scheduler.get_last_lr()[0]
        print(f"\n{phase_name} epoch {epoch}/{epochs} | lr={lr:.2e}")
        train_loss, train_acc = train_epoch(model, ema_model, loader, criterion, optimizer, ema_decay)
        _, val_acc, _, _ = evaluate(model, val_loader, criterion)
        _, ema_acc, _, _ = evaluate(ema_model, val_loader, criterion)
        scheduler.step()
        history.append({
            "epoch": epoch_offset + epoch,
            "phase": phase_name,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "ema_val_acc": ema_acc,
            "lr": lr
        })

        print(f"train acc: {train_acc:.5f} | val acc: {val_acc:.5f} | ema val acc: {ema_acc:.5f}")
        improved = False
        # salvez cel mai bun model normal
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), model_path)
            improved = True
        # salvez cel mai bun model EMA
        if ema_acc > best_ema:
            best_ema = ema_acc
            torch.save(ema_model.state_dict(), ema_path)
            improved = True
        bad_epochs = 0 if improved else bad_epochs + 1
        if bad_epochs >= patience:
            print("early stopping")
            break

    return best_val, best_ema

@torch.no_grad()
def tta_logits(model, images):
    big_images = F.interpolate(images, scale_factor=1.08, mode="bilinear", align_corners=False)
    return (
        0.55 * model(images)
        + 0.25 * model(torch.flip(images, dims=[3]))
        + 0.12 * model(big_images)
        + 0.08 * model(torch.flip(big_images, dims=[3]))
    )

@torch.no_grad()
def get_val_probs(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    for images, labels in loader:
        probs = torch.softmax(tta_logits(model, images), dim=1)
        probs_all.append(probs.detach().numpy())
        labels_all.extend(labels.numpy())
    return np.vstack(probs_all).astype(np.float32), np.array(labels_all)

@torch.no_grad()
def get_test_probs(model, loader):
    model.eval()
    ids_all, probs_all = [], []
    for images, image_ids in loader:
        probs = torch.softmax(tta_logits(model, images), dim=1)
        probs_all.append(probs.detach().numpy())
        ids_all.extend(list(image_ids))
    return ids_all, np.vstack(probs_all).astype(np.float32)

def save_confusion_matrix_plot(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    labels = [str(i) for i in range(1, num_classes + 1)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

def save_model_comparison_plot(names, values, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, values)
    ax.set_title("Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.70, max(values) + 0.02)
    for i, value in enumerate(values):
        ax.text(i, value + 0.002, f"{value:.4f}", ha="center")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

model = SignalCNN(num_classes)
ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad_(False)
history = []

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.035)
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay_initial)
scheduler = WarmupCosineLR(optimizer, warmup_epochs=warmup_epochs, total_epochs=initial_epochs)
best_val, best_ema = run_training_phase(
    "initial",
    model,
    ema_model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    initial_epochs,
    patience=16,
    ema_decay=0.995,
    best_val=0.0,
    best_ema=0.0,
    epoch_offset=0,
    history=history
)

initial_best_val = max(row["val_acc"] for row in history if row["phase"] == "initial")
initial_best_ema = max(row["ema_val_acc"] for row in history if row["phase"] == "initial")
print(f"\nbest initial val: {initial_best_val:.5f}")
print(f"best initial ema: {initial_best_ema:.5f}")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
ema_model.load_state_dict(torch.load(ema_path, map_location="cpu"))
ema_model.eval()

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.015)
optimizer = torch.optim.AdamW(model.parameters(), lr=fine_lr, weight_decay=weight_decay_fine)
scheduler = WarmupCosineLR(optimizer, warmup_epochs=2, total_epochs=fine_tune_epochs, eta_min=1e-7)

best_val, best_ema = run_training_phase(
    "fine",
    model,
    ema_model,
    train_fine_loader,
    criterion,
    optimizer,
    scheduler,
    fine_tune_epochs,
    patience=9,
    ema_decay=0.997,
    best_val=best_val,
    best_ema=best_ema,
    epoch_offset=initial_epochs,
    history=history
)

print(f"\nbest overall val: {best_val:.5f}")
print(f"best overall ema: {best_ema:.5f}")

pd.DataFrame(history).to_csv(history_path, index=False)

normal_model = SignalCNN(num_classes)
normal_model.load_state_dict(torch.load(model_path, map_location="cpu"))
normal_model.eval()
_, normal_acc, _, _ = evaluate(normal_model, val_loader, criterion)

ema_eval_model = SignalCNN(num_classes)
ema_eval_model.load_state_dict(torch.load(ema_path, map_location="cpu"))
ema_eval_model.eval()
_, ema_no_tta_acc, _, _ = evaluate(ema_eval_model, val_loader, criterion)

val_probs, y_val = get_val_probs(ema_eval_model, val_loader)
tta_preds = val_probs.argmax(axis=1)
ema_tta_acc = accuracy_score(y_val, tta_preds)

print("\nfinal validation results")
print(f"initial cnn : {initial_best_val:.5f}")
print(f"initial ema : {initial_best_ema:.5f}")
print(f"fine-tuned cnn : {normal_acc:.5f}")
print(f"ema no tta : {ema_no_tta_acc:.5f}")
print(f"ema + tta : {ema_tta_acc:.5f}")
print(f"tta gain : {ema_tta_acc - ema_no_tta_acc:+.5f}")

print("\nconfusion matrix:")
print(confusion_matrix(y_val, tta_preds, labels=np.arange(num_classes)))

comparison_names = ["Initial CNN", "Initial EMA", "Fine-tuned CNN", "EMA no TTA", "EMA + TTA"]
comparison_values = [initial_best_val, initial_best_ema, normal_acc, ema_no_tta_acc, ema_tta_acc]
save_model_comparison_plot(comparison_names, comparison_values, comparison_plot_path)
save_confusion_matrix_plot(y_val, tta_preds, confusion_matrix_path)

test_ids, test_probs = get_test_probs(ema_eval_model, test_loader)
test_preds = test_probs.argmax(axis=1) + 1

submission = pd.DataFrame({
    "id": test_ids,
    "label": test_preds.astype(int)
})
submission.to_csv(submission_path, index=False)

print("\nfinal summary")
print(f"submission : {submission_path}")
print(f"model normal : {model_path}")
print(f"model ema : {ema_path}")
print(f"history : {history_path}")
print(f"confusion matrix: {confusion_matrix_path}")
print(f"comparison plot : {comparison_plot_path}")
