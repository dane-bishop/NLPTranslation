from laser_encoders import LaserEncoderPipeline
encoder = LaserEncoderPipeline(lang="eng_Latn")
embeddings = encoder.encode_sentences(["Hi!", "This is a sentence encoder."])
print(embeddings.shape)  # (2, 1024)