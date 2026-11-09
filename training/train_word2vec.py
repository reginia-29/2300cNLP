"""Train a word2vec model on text8 and export it for the browser.

Usage:
    pip install gensim numpy
    curl -L -o text8.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
    python training/train_word2vec.py text8.gz

Writes:
    model/vectors.bin  - int8 matrix (VOCAB x DIM), each row an L2-normalised vector * 127
    model/meta.json    - vocabulary (most frequent first), counts and training settings
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else "text8.gz")
OUT = Path(__file__).resolve().parent.parent / "model"
EXPORT_VOCAB = 30000  # most frequent words shipped to the browser

SETTINGS = dict(
    vector_size=100,  # embedding dimension
    window=5,         # context words on each side
    min_count=5,      # ignore rarer words
    sg=1,             # 1 = skip-gram, 0 = CBOW
    negative=10,      # negative samples per positive pair
    sample=1e-4,      # down-sample very frequent words
    epochs=5,
    workers=4,
    seed=42,
)

if CORPUS.suffix == ".gz":
    raw = CORPUS.with_suffix("")
    if not raw.exists():
        with gzip.open(CORPUS, "rb") as src, open(raw, "wb") as dst:
            dst.write(src.read())
    CORPUS = raw

t0 = time.time()
model = Word2Vec(Text8Corpus(str(CORPUS)), **SETTINGS)
train_seconds = round(time.time() - t0)
wv = model.wv
print(f"trained in {train_seconds}s, vocab {len(wv)}")

words = wv.index_to_key[:EXPORT_VOCAB]  # gensim sorts by frequency
vecs = np.stack([wv[w] for w in words]).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)

OUT.mkdir(exist_ok=True)
q.tofile(OUT / "vectors.bin")
meta = {
    "corpus": "text8 (first 100 MB of English Wikipedia, cleaned; ~17M tokens)",
    "algorithm": "word2vec skip-gram with negative sampling (gensim)",
    "settings": {k: v for k, v in SETTINGS.items() if k not in ("workers", "seed")},
    "full_vocab": len(wv),
    "dim": SETTINGS["vector_size"],
    "train_seconds": train_seconds,
    "words": words,
    "counts": [int(wv.get_vecattr(w, "count")) for w in words],
}
(OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

for probe in ["king", "computer", "france", "music"]:
    print(probe, [w for w, _ in wv.most_similar(probe, topn=10)])
print("king - man + woman =", wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5))
