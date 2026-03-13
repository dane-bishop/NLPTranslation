from laser_encoders import LaserEncoderPipeline
import numpy as np
from pathlib import Path


# python -m pip install "datasets"
from datasets import load_dataset

dataset = load_dataset("allenai/nllb", "deu_Latn-eng_Latn", streaming=True, trust_remote_code=True)
encoder = LaserEncoderPipeline(lang="deu_Latn")
sentences = []
for i, row in enumerate(dataset["train"]):
	sentences.append(row["translation"]["deu_Latn"])
	if i % 50 == 0:
		print(f"Processed {i+1} rows", flush=True)

embeddings = encoder.encode_sentences(encode_sentences)

# Convert to one array: shape (N, 1024)
embeddings_array = np.vstack(embeddings)

# Save in same directory as this script
out_path = Path(__file__).parent / "callie_german.npy"
np.save(out_path, embeddings_array)

print(f"Saved embeddings to: {out_path}", flush=True)
print(f"Final shape: {embeddings_array.shape}", flush=True)