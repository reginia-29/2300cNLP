# Steady

Static site, deployed on Netlify (no build step; see `netlify.toml`).

| Page | What it is |
| --- | --- |
| `index.html` | Landing page |
| `steady.html` | Daily check-in app |
| `word2vec.html` | **Word explorer**: a trained word2vec model, 10 nearest words, analogies, 3D view |

## Word explorer (NLP)

- **Model:** word2vec skip-gram with negative sampling, trained with gensim on
  `text8` (first 100 MB of English Wikipedia, ~17M tokens).
  Settings: 100 dimensions, window 5, 10 negative samples, min_count 5, 5 epochs.
- **Files:** `model/vectors.bin` (30,000 most frequent words × 100 dims, int8,
  each row an L2-normalised vector × 127) and `model/meta.json` (vocabulary and settings).
- **Nearest words:** cosine similarity against every word, computed in the browser.
- **Analogy:** `a − b + c` (e.g. king − man + woman ≈ queen).
- **3D view:** PCA of the query and its neighbors (optionally their neighbors too),
  drawn on a canvas you can drag, zoom, and click.

Retrain:

```bash
pip install gensim numpy
curl -L -o text8.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
python training/train_word2vec.py text8.gz
```

Local preview (the page fetches the model, so serve over http):

```bash
python3 -m http.server 8000
```
