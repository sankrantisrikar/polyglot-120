# Polyglot-120

Multilingual language identification across 120+ languages, built on a
fine-tuned transformer, evaluated on clean, short, noisy, code-mixed and
mixed-script text, and served through a FastAPI endpoint with
confidence-based fallback routing.

## Results

Evaluated on the FLORES-200 devtest split: 121,440 held-out sentences, 120 languages, 29 scripts
(full tables in [`reports/REPORT.md`](reports/REPORT.md), raw numbers in `reports/metrics.json`).

| Slice | N | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| clean | 121,440 | **0.944** | 0.948 |
| short (2-5 words) | 36,000 | 0.776 | 0.782 |
| noisy (character-level corruption) | 36,000 | 0.728 | 0.748 |
| code-mixed (70/30 two languages) | 36,000 | 0.753 | 0.761 |
| mixed-script (70/30, different scripts) | 36,000 | 0.756 | 0.761 |

Per language (clean): median F1 0.999; 103 of 120 languages above 0.95 F1, 91 above 0.99,
39 at a perfect 1.000 (English 0.9995, German 0.9995, Mandarin Simplified 0.998, Hindi 0.991).
Excluding the 13 languages that belong to the hard pairs below, macro-F1 is 0.993.

**Where it fails.** 97% of the top-20 confusion pairs are within the same script, and 95% of
them are four families of near-identical written varieties: Arabic dialects (MSA vs Ta'izzi-Adeni,
Mesopotamian, Levantine, Tunisian, Egyptian, Moroccan; Ta'izzi-Adeni is never predicted at all,
every sentence goes to MSA), Cantonese vs Mandarin in Traditional script (713 of 1,012 Cantonese
sentences labelled Mandarin), Bosnian vs Croatian, and Indonesian vs Malay. These are the pairs
where sentence-level written text often carries no distinguishing signal.

**Confidence routing.** Because the errors cluster, the model's own confidence separates them well:

| Threshold | Coverage | Acc on kept | Acc on routed |
|---:|---:|---:|---:|
| 0.5 | 94.4% | 98.1% | 38.7% |
| 0.7 | 90.9% | 99.4% | 49.0% |
| **0.8** | **90.5%** | **99.5%** | 50.1% |
| 0.9 | 89.1% | 99.8% | 54.5% |
| 0.95 | 85.3% | 99.9% | 65.4% |

The deployed service uses `CONF_THRESHOLD=0.8`: 90.5% of inputs are answered directly at 99.5%
accuracy, the rest are routed to the fastText `lid.176` fallback, which returns its coarser label
(usually the macro-language: `ar`, `zh`, `hr`) instead of guessing the variety. Every response says
which path answered.

**Serving** (`scripts/bench.py` against the live API, 2 vCPU container, client in Texas, server in us-east):

| Load | Server-side p50 / p95 | Round trip p50 / p95 | Throughput |
|---|---|---|---:|
| 1 client, sequential | 77 ms / 93 ms | 337 ms / 425 ms | 2.8 req/s |
| 8 concurrent clients | 121 ms / 155 ms | 469 ms / 701 ms | 16.0 req/s |

Server-side is tokenization + forward pass + routing inside the API; round trip adds network and
queueing. Cold start after idle: about 15 s (weights load from the image, no download).

Live demo: [polyglot-120.vercel.app](https://polyglot-120.vercel.app) · API: [ssrikar1503--polyglot-120.us-east.modal.run](https://ssrikar1503--polyglot-120.us-east.modal.run/docs) (`/docs` for the OpenAPI UI, `/health` for status) · Code: [github.com/sankrantisrikar/polyglot-120](https://github.com/sankrantisrikar/polyglot-120)

---

## How it works

1. **Data** (`scripts/prepare_data.py`): FLORES-200 dev split for training,
   devtest for testing. The label set is the ~70 most widely used languages
   (English, Chinese, Hindi, Spanish, Arabic, ...) plus round-robin picks across
   scripts up to 120, so it is both useful and not Latin-heavy. Four robustness
   slices are synthesized from devtest.
2. **Model** (`scripts/train.py`): `xlm-roberta-base` fine-tuned for sequence
   classification, max length 64, one epoch.
3. **Evaluation** (`scripts/evaluate.py`): per-language P/R/F1, macro-F1 per
   slice, top confusion pairs, and a confidence-vs-accuracy table used to set
   the routing threshold.
4. **Serving** (`app/main.py`): FastAPI. If the top probability is below
   `CONF_THRESHOLD`, the request is routed to a fallback (fastText `lid.176`
   when present, otherwise `und`). Every response reports the path taken and
   latency.
5. **Benchmark** (`scripts/bench.py`): p50/p95/p99 latency and throughput
   against the live endpoint.

## Run it

### 0. Train on a free GPU (Google Colab, T4)

```bash
git clone <your-repo-url> && cd polyglot-120
pip install -r requirements-train.txt
huggingface-cli login            # only needed to upload your model; FLORES-200 downloads without an account

python scripts/prepare_data.py --out data --num_langs 120
python scripts/train.py --data data --out model --epochs 1
python scripts/evaluate.py --data data --model model --out reports
cat reports/REPORT.md            # copy the tables into this README
```

Expected time on a T4: data prep ~2 min, training ~25 min, evaluation ~5 min.
Or open `Polyglot120_Colab.ipynb` in Google Colab and Run all; it does steps 0 and 1 for you.

### 1. Push the model to the Hugging Face Hub

```bash
huggingface-cli repo create polyglot-120 --type model
huggingface-cli upload <your-hf-username>/polyglot-120 ./model .
```

### 2. Deploy the API on Modal (free, scales to zero)

```bash
pip install modal
modal setup                     # one-time browser login
modal deploy modal_app.py       # from the project root; prints the public URL
```

`modal_app.py` builds a CPU image with the model and the fastText fallback baked in, and
serves `app/main.py` at `https://<your-workspace>--polyglot-120.<region>.modal.run` (the exact URL is printed at the end of the deploy). Containers
scale to zero after 5 idle minutes; a cold start loads the weights from disk (10 to 30 s).
Change `HF_MODEL` or `CONF_THRESHOLD` at the top of the file and redeploy.

### 2b. Or: deploy the API as a Hugging Face Space (Docker)

1. On huggingface.co: New Space, name `polyglot-120`, SDK **Docker**, hardware **CPU basic** (free).
2. Push these files to the Space repo: `Dockerfile`, `requirements-serve.txt`,
   `app/`, and `space_README.md` renamed to `README.md`.
3. In the Space settings, add a **Variable** `MODEL_ID` = `<your-hf-username>/polyglot-120`
   and `CONF_THRESHOLD` = `0.8` (picked from the routing table in `reports/REPORT.md`).
4. Wait for the build. Your API is at `https://<your-hf-username>-polyglot-120.hf.space`.
   Check `/health` and `/docs`. Note: brand-new free accounts may get a `cpu-basic quota`
   403; paid `CPU upgrade` hardware ($0.03/h, sleeps when idle) or Modal above avoids it.

```bash
git clone https://huggingface.co/spaces/<your-hf-username>/polyglot-120 space
cp Dockerfile requirements-serve.txt space/ && cp -r app space/ && cp space_README.md space/README.md
cd space && git add . && git commit -m "Polyglot-120 API" && git push
```

### 3. Benchmark the live endpoint

```bash
pip install httpx
python scripts/bench.py --url https://<your-api-url> --n 300 --concurrency 8
```

Put the p50/p95 numbers in this README.

### 4. Deploy the frontend to Vercel

1. Open `web/index.html` and set `API_URL` to your API URL (Modal or Space, no trailing slash).
2. Push the repo to GitHub, then on vercel.com: Add New Project, import the repo,
   set **Root Directory** to `web`, Framework Preset **Other**, no build command, Deploy.
   You get `https://polyglot-120-<something>.vercel.app`.

Or from the command line:

```bash
npm i -g vercel
cd web && vercel --prod        # follow the prompts; framework = Other, no build command
```

### 5. Local run (optional)

```bash
pip install -r requirements-serve.txt
MODEL_ID=./model uvicorn app.main:app --port 7860
# then open web/index.html with API_URL set to http://localhost:7860
```

## Honest limitations

- Training data is FLORES-200 (news/wiki-style sentences, ~1,000 per language), so
  performance on very informal text is bounded by that domain. The robustness slices are
  synthesized from devtest, not collected from real users.
- Written Arabic dialects, Cantonese vs Mandarin in Traditional script, Bosnian vs Croatian and
  Indonesian vs Malay are largely indistinguishable at the sentence level in this register. The
  model does not solve that; the confidence router makes the failure explicit and falls back to
  a macro-language label.
- 2 to 5 word inputs lose about 17 points of macro-F1, and character-level noise about 22. Short-text LangID needs
  character n-gram features or more short-text training data; neither is done here.
- Tifinagh (Central Atlas Tamazight) and Ol Chiki (Santali) are confused with each other
  151 times despite sharing nothing; both scripts sit outside XLM-R's pretraining
  coverage, so they likely tokenize to mostly unknown tokens. A byte-level or
  character n-gram feature would fix this.
- One epoch, one backbone, no hyperparameter search. Numbers are a baseline, not a ceiling.
- No Common Crawl data is used.
