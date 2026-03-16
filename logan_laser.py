from datasets import load_dataset
from laser_encoders import LaserEncoderPipeline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from tqdm import tqdm
import mplcursors

dataset = load_dataset("allenai/nllb", "deu_Latn-eng_Latn", streaming=True)

eng_encoder = LaserEncoderPipeline(lang="eng_Latn")
deu_encoder = LaserEncoderPipeline(lang="deu_Latn")

eng_embeddings = []
deu_embeddings = []
eng_sentences = []
deu_sentences = []

N = 1000

for i, row in tqdm(enumerate(dataset["train"]), total=N):
    eng = row["translation"]["eng_Latn"]
    deu = row["translation"]["deu_Latn"]

    eng_sentences.append(eng)
    deu_sentences.append(deu)

    eng_embeddings.append(eng_encoder.encode_sentences([eng])[0])
    deu_embeddings.append(deu_encoder.encode_sentences([deu])[0])

    if i + 1 == N:
        break

eng_embs = np.stack(eng_embeddings)
deu_embs = np.stack(deu_embeddings)

X = np.vstack([eng_embs, deu_embs])

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

eng_2d = X_2d[:len(eng_embs)]
deu_2d = X_2d[len(eng_embs):]

fig, ax = plt.subplots(figsize=(10, 8))

# Draw faint connection lines
segments = [
    [(eng_2d[i, 0], eng_2d[i, 1]), (deu_2d[i, 0], deu_2d[i, 1])]
    for i in range(N)
]
lc = LineCollection(segments, linewidths=0.7, alpha=0.15, zorder=1)
ax.add_collection(lc)

# Draw points above lines
sc_eng = ax.scatter(eng_2d[:, 0], eng_2d[:, 1], s=20, alpha=0.7, label="English", zorder=2)
sc_deu = ax.scatter(deu_2d[:, 0], deu_2d[:, 1], s=20, alpha=0.7, label="German", zorder=2)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("English vs German PCA with translation-pair links")
ax.legend()

cursor = mplcursors.cursor([sc_eng, sc_deu], hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    if sel.artist == sc_eng:
        sel.annotation.set_text(
            f"[ EN ] {eng_sentences[idx]}\n\n[ DE ] {deu_sentences[idx]}"
        )
    else:
        sel.annotation.set_text(
            f"[ DE ] {deu_sentences[idx]}\n\n[ EN ] {eng_sentences[idx]}"
        )

plt.tight_layout()
plt.show()