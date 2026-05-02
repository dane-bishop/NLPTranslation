import os
import threading
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import load_state_dict as _load_state_dict_from_file


MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")

_MODEL = None
_TOKENIZER = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOCK = threading.Lock()


def _load_state_dict_cpu(path: str, is_safetensors: bool) -> dict:
    """Load state dict to CPU. Use torch.load for .bin to avoid transformers' torch>=2.6 requirement."""
    if is_safetensors:
        return _load_state_dict_from_file(path, map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_model_without_meta() -> AutoModelForSeq2SeqLM:
    """Load model by creating from config then loading state dict with map_location='cpu'.
    Avoids meta tensors so .to(device) works. Tries safetensors then PyTorch .bin (NLLB uses .bin).
    """
    import json

    from transformers.utils import cached_file

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_config(config)

    # Try possible weight filenames (repo may have safetensors or pytorch_model.bin)
    candidates = [
        ("model.safetensors", True),
        ("model.safetensors.index.json", True),
        ("pytorch_model.bin", False),
        ("pytorch_model.bin.index.json", False),
    ]
    state_dict = None
    for name, is_safetensors in candidates:
        try:
            weight_file = cached_file(MODEL_NAME, name)
            if name.endswith(".json"):
                with open(weight_file) as f:
                    index = json.load(f)
                state_dict = {}
                for shard_name in set(index.get("weight_map", {}).values()):
                    shard_path = cached_file(MODEL_NAME, shard_name)
                    state_dict.update(
                        _load_state_dict_cpu(shard_path, shard_path.endswith(".safetensors"))
                    )
            else:
                state_dict = _load_state_dict_cpu(weight_file, is_safetensors)
            break
        except OSError:
            continue

    if state_dict is None:
        raise OSError(
            f"Could not find model weights for {MODEL_NAME}. "
            "Tried: model.safetensors, model.safetensors.index.json, pytorch_model.bin, pytorch_model.bin.index.json"
        )

    model.load_state_dict(state_dict, strict=False, assign=True)
    model.tie_weights()
    return model


def _ensure_model_loaded() -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    global _MODEL, _TOKENIZER
    with _LOCK:
        if _TOKENIZER is None:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

        if _MODEL is None:
            # Always use config + state-dict load. from_pretrained + .to(device) often leaves
            # some tensors on meta (e.g. tied weights), causing "Cannot copy out of meta tensor".
            _MODEL = _load_model_without_meta()
            _MODEL = _MODEL.to(_DEVICE)
            _MODEL.eval()

    return _MODEL, _TOKENIZER


def _pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled


def _clean_tokens(tokens: list[str]) -> list[str]:
    cleaned = []
    for token in tokens:
        text = token.replace("▁", " ").strip()
        if not text:
            text = token
        cleaned.append(text)
    return cleaned


def _token_alignment_fallback(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    source_text: str,
    source_lang: str,
    target_text: str,
    target_lang: str,
) -> dict[str, Any]:
    tokenizer.src_lang = source_lang
    encoded_source = tokenizer(source_text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        source_encoder = model.model.encoder(
            input_ids=encoded_source["input_ids"],
            attention_mask=encoded_source["attention_mask"],
            return_dict=True,
        )

    tokenizer.src_lang = target_lang
    encoded_target = tokenizer(target_text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        target_encoder = model.model.encoder(
            input_ids=encoded_target["input_ids"],
            attention_mask=encoded_target["attention_mask"],
            return_dict=True,
        )

    src_hidden = source_encoder.last_hidden_state[0]
    tgt_hidden = target_encoder.last_hidden_state[0]
    src_hidden = src_hidden / src_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tgt_hidden = tgt_hidden / tgt_hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    matrix = (tgt_hidden @ src_hidden.T).detach().cpu().numpy().tolist()

    source_tokens = tokenizer.convert_ids_to_tokens(encoded_source["input_ids"][0])
    target_tokens = tokenizer.convert_ids_to_tokens(encoded_target["input_ids"][0])
    return {
        "source_tokens": _clean_tokens(source_tokens),
        "target_tokens": _clean_tokens(target_tokens),
        "values": matrix,
    }


def translate_text(text: str, source_lang: str, target_lang: str, max_new_tokens: int = 96) -> str:
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = source_lang

    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
        )
    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translated


def embed_text(text: str, lang: str) -> np.ndarray:
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        encoder_out = model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
    pooled = _pool_hidden(encoder_out.last_hidden_state, encoded["attention_mask"]).squeeze(0)
    return pooled.detach().cpu().numpy()


DEFAULT_CONTEXT_TEMPLATE = "I saw a {word} near the river."


def _find_word_token_indices(
    input_ids: torch.Tensor, tokenizer, word: str
) -> "list[int] | None":
    """Find token indices for a target word within a tokenized sentence.

    Tries exact subsequence match first, then falls back to
    reconstructing token text piece-by-piece.
    """
    word_token_ids = tokenizer.encode(word, add_special_tokens=False)
    all_ids = input_ids[0].tolist()

    n = len(word_token_ids)
    for i in range(len(all_ids) - n + 1):
        if all_ids[i : i + n] == word_token_ids:
            return list(range(i, i + n))

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_lower = word.lower()

    for start in range(len(tokens)):
        reconstructed = ""
        for end in range(start, len(tokens)):
            piece = tokens[end].replace("\u2581", "").replace("\u0120", "")
            reconstructed += piece
            if reconstructed.lower() == word_lower:
                return list(range(start, end + 1))
            if len(reconstructed) > len(word) * 2:
                break

    return None


def embed_word_in_context(
    word: str,
    lang: str,
    context_template: str = DEFAULT_CONTEXT_TEMPLATE,
) -> np.ndarray:
    """Embed a word inside a carrier sentence, returning only the target word's token activations.

    Provides contextual disambiguation while isolating the
    concept-specific representation from carrier-sentence noise.
    Falls back to full mean-pooling if the word tokens cannot be located.
    """
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    sentence = context_template.format(word=word)
    encoded = tokenizer(sentence, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        encoder_out = model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )

    hidden = encoder_out.last_hidden_state  # (1, seq_len, hidden_dim)
    word_indices = _find_word_token_indices(encoded["input_ids"], tokenizer, word)

    if word_indices is not None and len(word_indices) > 0:
        pooled = hidden[0, word_indices, :].mean(dim=0)
    else:
        pooled = _pool_hidden(hidden, encoded["attention_mask"]).squeeze(0)

    return pooled.detach().cpu().numpy()


def embed_word_in_context_all_layers(
    word: str,
    lang: str,
    context_template: str = DEFAULT_CONTEXT_TEMPLATE,
    *,
    include_embedding_layer: bool = False,
) -> list[np.ndarray]:
    """Embed a word in context and return pooled vectors for all encoder layers.

    By default this returns one vector per *encoder block* (12 for NLLB-200 distilled),
    excluding the initial token embedding output. Set include_embedding_layer=True
    to include that first hidden state as layer 0.
    """
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    sentence = context_template.format(word=word)
    encoded = tokenizer(sentence, return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        out = model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = list(out.hidden_states or [])
    if not hidden_states:
        # Fallback: just return final layer embedding (compatible with callers)
        return [embed_word_in_context(word, lang, context_template=context_template)]

    if not include_embedding_layer:
        hidden_states = hidden_states[1:]

    word_indices = _find_word_token_indices(encoded["input_ids"], tokenizer, word)

    pooled_per_layer: list[np.ndarray] = []
    for h in hidden_states:
        # h: (1, seq_len, dim)
        if word_indices is not None and len(word_indices) > 0:
            pooled = h[0, word_indices, :].mean(dim=0)
        else:
            pooled = _pool_hidden(h, encoded["attention_mask"]).squeeze(0)
        pooled_per_layer.append(pooled.detach().cpu().numpy())

    return pooled_per_layer


def embed_words_in_context_batch(
    words: list[str],
    langs: list[str],
    context_template: str = DEFAULT_CONTEXT_TEMPLATE,
) -> list[np.ndarray]:
    """Batch version of embed_word_in_context.

    Each word is placed into *context_template*, the full sentence is
    encoded, and only the target word's subword token hidden states
    are mean-pooled to produce the embedding.
    """
    model, tokenizer = _ensure_model_loaded()
    results: list[tuple[int, np.ndarray]] = []
    indexed = list(enumerate(zip(words, langs)))
    indexed.sort(key=lambda x: x[1][1])

    for idx, (word, lang) in indexed:
        tokenizer.src_lang = lang
        sentence = context_template.format(word=word)
        encoded = tokenizer(sentence, return_tensors="pt").to(_DEVICE)

        with torch.no_grad():
            out = model.model.encoder(**encoded, return_dict=True)

        hidden = out.last_hidden_state
        word_indices = _find_word_token_indices(encoded["input_ids"], tokenizer, word)

        if word_indices is not None and len(word_indices) > 0:
            vec = hidden[0, word_indices, :].mean(dim=0)
        else:
            vec = _pool_hidden(hidden, encoded["attention_mask"]).squeeze(0)

        results.append((idx, vec.detach().cpu().numpy()))

    results.sort(key=lambda x: x[0])
    return [vec for _, vec in results]


def sentence_similarity_matrix(vectors: list[np.ndarray]) -> list[list[float]]:
    array = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.clip(norms, 1e-8, None)
    matrix = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return matrix.tolist()


def _flag_embedding_outliers(
    array: np.ndarray, iqr_factor: float = 3.0
) -> np.ndarray:
    """Return boolean mask where True = inlier, False = outlier.

    Uses embedding L2 norms and IQR fencing to detect degenerate
    vectors (near-zero or extreme magnitude) that distort PCA.
    """
    norms = np.linalg.norm(array, axis=1)
    q1, q3 = np.percentile(norms, [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return (norms >= max(lower, 1e-6)) & (norms <= upper)


def project_embeddings(vectors: list[np.ndarray], labels: list[str]) -> list[dict[str, Any]]:
    n = len(vectors)
    if n < 2:
        return [{"label": labels[0], "x": 0.0, "y": 0.0, "z": 0.0}]
    array = np.array(vectors, dtype=np.float64)
    n_components = min(3, n, array.shape[1])

    inlier_mask = _flag_embedding_outliers(array)
    if inlier_mask.sum() >= max(n_components + 1, 4):
        pca = PCA(n_components=n_components).fit(array[inlier_mask])
        projected = pca.transform(array)
    else:
        projected = PCA(n_components=n_components).fit_transform(array)

    points = []
    for i, label in enumerate(labels):
        point: dict[str, Any] = {
            "label": label,
            "x": float(projected[i, 0]),
            "y": float(projected[i, 1]),
            "z": float(projected[i, 2]) if n_components >= 3 else 0.0,
        }
        points.append(point)
    return points


def cross_attention_map(
    source_text: str,
    source_lang: str,
    target_text: str,
    target_lang: str,
) -> dict[str, Any]:
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = source_lang

    encoded_source = tokenizer(source_text, return_tensors="pt").to(_DEVICE)
    encoded_target = tokenizer(target_text, return_tensors="pt").to(_DEVICE)
    decoder_input_ids = encoded_target["input_ids"][:, :-1]
    if decoder_input_ids.shape[1] == 0:
        return _token_alignment_fallback(
            model,
            tokenizer,
            source_text,
            source_lang,
            target_text,
            target_lang,
        )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded_source["input_ids"],
            attention_mask=encoded_source["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    cross_layers = [layer for layer in (outputs.cross_attentions or []) if layer is not None]
    if not cross_layers:
        return _token_alignment_fallback(
            model,
            tokenizer,
            source_text,
            source_lang,
            target_text,
            target_lang,
        )

    cross = torch.stack(cross_layers)  # layers, batch, heads, tgt, src
    matrix = cross.mean(dim=(0, 2))[0]  # tgt, src
    matrix = matrix.detach().cpu().numpy().tolist()

    source_tokens = tokenizer.convert_ids_to_tokens(encoded_source["input_ids"][0])
    target_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
    return {
        "source_tokens": _clean_tokens(source_tokens),
        "target_tokens": _clean_tokens(target_tokens),
        "values": matrix,
    }


def abtt_correct(vectors: list[np.ndarray], k: int = 3) -> list[np.ndarray]:
    array = np.array(vectors)
    array = array - array.mean(axis=0)
    pca = PCA(n_components=k)
    pca.fit(array)
    for component in pca.components_:
        array = array - (array @ component[:, None]) * component[None, :]
    return [array[i] for i in range(len(vectors))]


def sentence_similarity_matrix_corrected(vectors: list[np.ndarray], k: int = 3) -> list[list[float]]:
    corrected = abtt_correct(vectors, k=k)
    return sentence_similarity_matrix(corrected)


def project_embeddings_mean_centered(
    vectors: list[np.ndarray], labels: list[str], language_ids: list[str]
) -> dict[str, Any]:
    array = np.array(vectors, dtype=np.float64)
    n = len(vectors)
    n_components = min(3, n, array.shape[1])

    lang_set = sorted(set(language_ids))
    samples_per_lang = {
        lang: sum(1 for l in language_ids if l == lang) for lang in lang_set
    }
    degenerate = all(c <= 1 for c in samples_per_lang.values())

    if degenerate:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        centered = array / np.clip(norms, 1e-8, None)
        method = "cosine_pca"
    else:
        lang_means = {}
        for lang in lang_set:
            indices = [i for i, l in enumerate(language_ids) if l == lang]
            lang_means[lang] = array[indices].mean(axis=0)
        centered = np.array([
            array[i] - lang_means[language_ids[i]] for i in range(n)
        ])
        method = "mean_centered"

    inlier_mask = _flag_embedding_outliers(array)
    fit_on_inliers = inlier_mask.sum() >= max(n_components + 1, 4)

    if fit_on_inliers:
        raw_pca = PCA(n_components=n_components).fit(array[inlier_mask])
        raw_proj = raw_pca.transform(array)
        centered_pca = PCA(n_components=n_components).fit(centered[inlier_mask])
        centered_proj = centered_pca.transform(centered)
    else:
        raw_proj = PCA(n_components=n_components).fit_transform(array)
        centered_proj = PCA(n_components=n_components).fit_transform(centered)

    def _to_points(proj, lbls):
        points = []
        for i, label in enumerate(lbls):
            points.append({
                "label": label,
                "x": float(proj[i, 0]),
                "y": float(proj[i, 1]),
                "z": float(proj[i, 2]) if n_components >= 3 else 0.0,
            })
        return points

    return {
        "raw": _to_points(raw_proj, labels),
        "centered": _to_points(centered_proj, labels),
        "method": method,
    }


def embed_text_all_layers(text: str, lang: str) -> list[np.ndarray]:
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        out = model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    return [
        _pool_hidden(h, encoded["attention_mask"]).squeeze(0).cpu().numpy()
        for h in out.hidden_states
    ]


def embed_text_batch(texts: list[str], langs: list[str]) -> list[np.ndarray]:
    model, tokenizer = _ensure_model_loaded()
    results: list[tuple[int, np.ndarray]] = []
    indexed = list(enumerate(zip(texts, langs)))
    indexed.sort(key=lambda x: x[1][1])

    for idx, (text, lang) in indexed:
        tokenizer.src_lang = lang
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(_DEVICE)
        with torch.no_grad():
            out = model.model.encoder(**encoded, return_dict=True)
        vec = _pool_hidden(out.last_hidden_state, encoded["attention_mask"]).squeeze(0).cpu().numpy()
        results.append((idx, vec))

    results.sort(key=lambda x: x[0])
    return [vec for _, vec in results]


def cross_attention_map_per_head(
    source_text: str,
    source_lang: str,
    target_text: str,
    target_lang: str,
) -> dict[str, Any]:
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = source_lang
    encoded_source = tokenizer(source_text, return_tensors="pt").to(_DEVICE)
    encoded_target = tokenizer(target_text, return_tensors="pt").to(_DEVICE)
    decoder_input_ids = encoded_target["input_ids"][:, :-1]

    if decoder_input_ids.shape[1] == 0:
        return _token_alignment_fallback(
            model, tokenizer, source_text, source_lang, target_text, target_lang
        )

    with torch.no_grad():
        outputs = model(
            input_ids=encoded_source["input_ids"],
            attention_mask=encoded_source["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    cross_layers = [layer for layer in (outputs.cross_attentions or []) if layer is not None]
    if not cross_layers:
        return _token_alignment_fallback(
            model, tokenizer, source_text, source_lang, target_text, target_lang
        )

    per_head_layers = []
    for layer_idx, layer_attn in enumerate(cross_layers):
        heads = layer_attn[0]  # (num_heads, tgt_len, src_len)
        confidence = heads.max(dim=-1).values.mean(dim=-1)  # (num_heads,)
        per_head_layers.append({
            "layer": layer_idx,
            "head_confidence": confidence.cpu().numpy().tolist(),
            "attention": heads.cpu().numpy().tolist(),
        })

    source_tokens = tokenizer.convert_ids_to_tokens(encoded_source["input_ids"][0])
    target_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
    return {
        "source_tokens": _clean_tokens(source_tokens),
        "target_tokens": _clean_tokens(target_tokens),
        "per_head_layers": per_head_layers,
    }


def attention_entropy(attn_values: list[list[float]]) -> float:
    mat = np.array(attn_values)
    mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-8, None)
    H = -np.sum(mat * np.log(mat + 1e-10), axis=1)
    return float(H.mean())


def neuron_activation_mask(text: str, lang: str, threshold_pct: float = 0.9) -> np.ndarray:
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)

    activations = {}
    def hook_fn(module, input, output):
        activations["ffn"] = output.detach()

    last_ffn = model.model.encoder.layers[-1].fc2
    handle = last_ffn.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model.model.encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
    finally:
        handle.remove()

    act = activations["ffn"].squeeze(0)  # (seq_len, hidden_dim)
    max_act = act.abs().max(dim=0).values.cpu().numpy()  # (hidden_dim,)
    threshold = np.percentile(max_act, threshold_pct * 100)
    mask = (max_act >= threshold).astype(np.int8)
    return mask


def neuron_overlap_matrix(
    texts: list[str], langs: list[str], threshold_pct: float = 0.9
) -> list[list[float]]:
    masks = [neuron_activation_mask(text, lang, threshold_pct) for text, lang in zip(texts, langs)]
    n = len(masks)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            intersection = np.sum(masks[i] & masks[j])
            union = np.sum(masks[i] | masks[j])
            matrix[i][j] = float(intersection / max(union, 1))
    return matrix
