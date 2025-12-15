import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA


DATA_ROOT = "att_faces"
NON_FACE_DIR = "non_faces"
FIG_DIR = "figures"
SEED = 42

os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device=device
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


def load_faces(root):
    paths, labels = [], []
    for subj in sorted(os.listdir(root)):
        d = os.path.join(root, subj)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".pgm"):
                paths.append(os.path.join(d, f))
                labels.append(subj)
    return np.array(paths), np.array(labels)


def load_non_faces(root, max_n=40, seed=42):
    rng = np.random.default_rng(seed)
    paths = [
        os.path.join(root, f)
        for f in sorted(os.listdir(root))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    rng.shuffle(paths)
    return np.array(paths[:max_n])


face_paths, face_labels = load_faces(DATA_ROOT)
non_face_paths = load_non_faces(NON_FACE_DIR, max_n=40, seed=SEED)

print("Faces:", len(face_paths))
print("Non-faces:", len(non_face_paths))

def mtcnn_face_stats(paths, name=""):
    n_face = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        if mtcnn(img) is not None:
            n_face += 1
    print(f"\n{name}")
    print(f"  Total: {len(paths)}")
    print(f"  Faces detected: {n_face}")
    print(f"  Rate: {n_face / len(paths):.3f}")


mtcnn_face_stats(face_paths, "ATT Faces")
mtcnn_face_stats(non_face_paths, "Non-Faces")


def per_class_split(paths, labels, seed=42):
    rng = np.random.default_rng(seed)
    tr_p, va_p, te_p = [], [], []
    tr_y, va_y, te_y = [], [], []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        cls_paths = paths[idx]
        rng.shuffle(cls_paths)

        va_p.append(cls_paths[0])
        te_p.append(cls_paths[1])

        for p in cls_paths[2:]:
            tr_p.append(p)
            tr_y.append(cls)

        va_y.append(cls)
        te_y.append(cls)

    return (
        np.array(tr_p), np.array(va_p), np.array(te_p),
        np.array(tr_y), np.array(va_y), np.array(te_y)
    )


train_p, val_p, test_p, train_y, val_y, test_y = \
    per_class_split(face_paths, face_labels, SEED)


@torch.no_grad()
def extract_embeddings(paths, labels=None):
    embs, out_labels = [], []

    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        face = mtcnn(img)
        if face is None:
            continue

        face = (face - 0.5) / 0.5
        face = face.unsqueeze(0).to(device)

        e = facenet(face).cpu().numpy()[0]
        e /= np.linalg.norm(e) + 1e-12

        embs.append(e)
        if labels is not None:
            out_labels.append(labels[i])

    return np.vstack(embs), np.array(out_labels)


X_train, y_train = extract_embeddings(train_p, train_y)
X_val, y_val = extract_embeddings(val_p, val_y)
X_test, y_test = extract_embeddings(test_p, test_y)


k_list = range(1, 9)
val_accs = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_val, knn.predict(X_val))
    val_accs.append(acc)
    print(f"k={k} | val acc={acc:.3f}")

best_k = k_list[np.argmax(val_accs)]
print("Best k:", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine")
knn.fit(X_train, y_train)


test_pred = knn.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print("TEST accuracy:", test_acc)

cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Faces)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix_faces.png")
plt.close()


plt.figure(figsize=(6,6))
plt.plot(list(k_list), val_accs, marker="o")
plt.axvline(best_k, linestyle="--", label=f"best k = {best_k}")
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("kNN Validation Accuracy vs k")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/knn_val_accuracy_vs_k.png")
plt.close()


pca = PCA(n_components=2, random_state=SEED)
Xtr_2d = pca.fit_transform(X_train)
Xte_2d = pca.transform(X_test)

nn = NearestNeighbors(n_neighbors=best_k, metric="cosine")
nn.fit(X_train)
_, nbr_idx = nn.kneighbors(X_test)

classes = np.unique(y_train)
cls_to_id = {c: i for i, c in enumerate(classes)}
ytr_id = np.array([cls_to_id[c] for c in y_train])
yte_id = np.array([cls_to_id[c] for c in y_test])

plt.figure(figsize=(7, 6))
plt.scatter(Xtr_2d[:, 0], Xtr_2d[:, 1], c=ytr_id, s=20, alpha=0.5)
plt.scatter(Xte_2d[:, 0], Xte_2d[:, 1], c=yte_id, s=80, marker="x")

n_show = min(5, len(X_test))
for i in range(n_show):
    for j in nbr_idx[i]:
        plt.plot(
            [Xte_2d[i, 0], Xtr_2d[j, 0]],
            [Xte_2d[i, 1], Xtr_2d[j, 1]],
            alpha=0.4
        )

plt.title(f"kNN Neighbor Links in PCA Space (k={best_k})")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/knn_neighbors_pca.png")
plt.close()


@torch.no_grad()
def extract_no_det(paths):
    embs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((160, 160))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = (img - 0.5) / 0.5
        img = img.unsqueeze(0).to(device)

        e = facenet(img).cpu().numpy()[0]
        e /= np.linalg.norm(e) + 1e-12
        embs.append(e)

    return np.vstack(embs)


X_face = extract_no_det(train_p[:100])
X_non = extract_no_det(non_face_paths)

X_all = np.vstack([X_face, X_non])
y_all = np.array(["Face"] * len(X_face) + ["Non-face"] * len(X_non))

pca = PCA(n_components=2, random_state=SEED)
X2 = pca.fit_transform(X_all)

plt.figure(figsize=(6, 6))
for lbl, col in [("Face", "blue"), ("Non-face", "red")]:
    idx = y_all == lbl
    plt.scatter(X2[idx, 0], X2[idx, 1], label=lbl, alpha=0.6)

plt.title("FaceNet Embeddings: Face vs Non-Face")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/pca_face_vs_nonface.png")
plt.close()


pca = PCA(n_components=2, random_state=SEED)
Xf = pca.fit_transform(X_train)

plt.figure(figsize=(6, 6))
for cls in np.unique(y_train):
    idx = y_train == cls
    plt.scatter(Xf[idx, 0], Xf[idx, 1], alpha=0.6)

plt.title("FaceNet Embeddings (Faces Only)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/pca_faces_only.png")
plt.close()

print("\nAll results saved to:", FIG_DIR)
