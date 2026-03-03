from laser_encoders import LaserEncoderPipeline

# python -m pip install "datasets"
from datasets import load_dataset

dataset = load_dataset("allenai/nllb", "deu_Latn-eng_Latn", streaming=True, trust_remote_code=True)
encoder = LaserEncoderPipeline(lang="eng_Latn")
embeddings = []
for i, row in enumerate(dataset["train"]):
	embeddings.append(encoder.encode_sentences(row["translation"]["deu_Latn"]))
	print(embeddings[0].shape)  # (2, 1024)
	break
