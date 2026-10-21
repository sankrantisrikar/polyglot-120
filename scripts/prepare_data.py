"""
Prepare training and evaluation data for Polyglot-120 from FLORES-200.

FLORES-200 gives ~1,000 parallel sentences per language for 200+ languages
in two splits: dev (used here for training) and devtest (used for testing).

Data source, in order:
  1. Meta's public tarball (no account, no gating):
       https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  2. Fallback: the Hugging Face dataset facebook/flores (gated; needs login
     and clicking "Agree" on the dataset page).

Outputs (in --out):
  train.csv              text,label           (dev split)
  test_clean.csv         text,label,slice     (devtest, unmodified)
  test_short.csv         truncated inputs (2-5 words, or 6-12 chars for unspaced scripts)
  test_noisy.csv         character-level noise: drops, swaps, duplicates, case flips
  test_code_mixed.csv    70% of a sentence in label language + 30% from another language
  test_mixed_script.csv  like code_mixed, but the other language uses a different script
  labels.json            sorted list of language codes

Language selection: every MUST_HAVE language that exists in FLORES-200 (about 70
widely used languages, English included) plus round-robin picks across scripts up
to --num_langs, so the set is both useful and script-diverse.

Usage:
  python scripts/prepare_data.py --out data --num_langs 120
  python scripts/prepare_data.py --out data --num_langs 0      # all languages
"""
import argparse
import csv
import io
import json
import os
import random
import tarfile
import urllib.request
from collections import defaultdict

TARBALL_URLS = [
    "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    "https://tinyurl.com/flores200dataset",
]

# Always included (when present in FLORES-200): the most widely used languages, so the
# label set is useful as a product, not just as a benchmark. The remaining slots are
# filled round-robin across scripts for diversity (see pick_languages).
MUST_HAVE = [
    "eng_Latn", "zho_Hans", "zho_Hant", "hin_Deva", "spa_Latn", "fra_Latn", "arb_Arab",
    "ben_Beng", "por_Latn", "rus_Cyrl", "urd_Arab", "ind_Latn", "deu_Latn", "jpn_Jpan",
    "swh_Latn", "mar_Deva", "tel_Telu", "tur_Latn", "tam_Taml", "vie_Latn", "kor_Hang",
    "ita_Latn", "pes_Arab", "pol_Latn", "ukr_Cyrl", "nld_Latn", "tha_Thai", "guj_Gujr",
    "kan_Knda", "mal_Mlym", "mya_Mymr", "pan_Guru", "ron_Latn", "hun_Latn", "ces_Latn",
    "ell_Grek", "swe_Latn", "heb_Hebr", "fin_Latn", "dan_Latn", "nob_Latn", "zsm_Latn",
    "tgl_Latn", "hau_Latn", "yor_Latn", "ibo_Latn", "amh_Ethi", "som_Latn", "khm_Khmr",
    "lao_Laoo", "sin_Sinh", "npi_Deva", "kat_Geor", "hye_Armn", "azj_Latn", "kaz_Cyrl",
    "uzn_Latn", "bul_Cyrl", "srp_Cyrl", "hrv_Latn", "slk_Latn", "slv_Latn", "lit_Latn",
    "lvs_Latn", "est_Latn", "cat_Latn", "afr_Latn", "jav_Latn", "ceb_Latn", "zul_Latn",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_from_tarball(cache_dir):
    """Download Meta's public FLORES-200 tarball and parse dev/devtest files.
    Returns {"dev": {lang: [sentences]}, "devtest": {lang: [sentences]}}."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, "flores200_dataset.tar.gz")
    if not os.path.exists(local):
        last_err = None
        for url in TARBALL_URLS:
            try:
                print(f"downloading {url} ...")
                urllib.request.urlretrieve(url, local)
                break
            except Exception as e:  # try next mirror
                last_err = e
                print("  failed:", e)
        if not os.path.exists(local):
            raise RuntimeError(f"could not download FLORES-200 tarball: {last_err}")
    splits = {"dev": defaultdict(list), "devtest": defaultdict(list)}
    with tarfile.open(local, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.replace("\\", "/")
            parts = name.split("/")
            if len(parts) < 2:
                continue
            split, fname = parts[-2], parts[-1]
            if split not in splits:
                continue
            if not (fname.endswith(".dev") or fname.endswith(".devtest")):
                continue
            lang = fname.rsplit(".", 1)[0]  # e.g. eng_Latn
            text = io.TextIOWrapper(tf.extractfile(m), encoding="utf-8").read()
            splits[split][lang] = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not splits["dev"] or not splits["devtest"]:
        raise RuntimeError("tarball parsed but no dev/devtest files found")
    print(f"tarball: {len(splits['dev'])} languages in dev, {len(splits['devtest'])} in devtest")
    return splits


def load_from_hub():
    """Fallback: gated Hub dataset. Requires `huggingface-cli login` and
    accepting the terms at https://huggingface.co/datasets/facebook/flores"""
    from datasets import load_dataset
    ds = load_dataset("facebook/flores", "all")
    splits = {"dev": defaultdict(list), "devtest": defaultdict(list)}
    for split in splits:
        cols = [c for c in ds[split].column_names if c.startswith("sentence_")]
        for row in ds[split]:
            for c in cols:
                t = row[c]
                if t and t.strip():
                    splits[split][c[len("sentence_"):]].append(t.strip())
    return splits


def load_flores(cache_dir):
    try:
        return load_from_tarball(cache_dir)
    except Exception as e:
        print("tarball route failed, trying Hugging Face Hub:", e)
        return load_from_hub()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def pick_languages(all_langs, n, rng, must_have=()):
    """Pick n languages: every available MUST_HAVE language first, then fill the
    remaining slots round-robin across scripts so the set isn't all Latin-script."""
    if n <= 0 or n >= len(all_langs):
        return sorted(all_langs)
    avail = set(all_langs)
    chosen = [l for l in must_have if l in avail][:n]
    missing = [l for l in must_have if l not in avail]
    if missing:
        print(f"note: {len(missing)} must-have languages not in this FLORES release: {missing}")
    by_script = defaultdict(list)
    for code in all_langs:
        if code not in chosen:
            by_script[code.split("_")[-1]].append(code)
    for v in by_script.values():
        rng.shuffle(v)
    scripts = sorted(by_script.keys())
    while len(chosen) < n:
        progressed = False
        for s in scripts:
            if by_script[s]:
                chosen.append(by_script[s].pop())
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            break
    return sorted(chosen)


def has_spaces(text):
    return " " in text.strip()


def make_short(text, rng):
    if has_spaces(text):
        words = text.split()
        return " ".join(words[: rng.randint(2, min(5, len(words)))])
    return text[: rng.randint(6, min(12, len(text)))]


def make_noisy(text, rng, p=0.08):
    chars, out, i = list(text), [], 0
    while i < len(chars):
        c, r = chars[i], rng.random()
        if r < p / 3:
            i += 1  # drop
            continue
        if r < 2 * p / 3 and i + 1 < len(chars):
            out.extend([chars[i + 1], c])  # swap
            i += 2
            continue
        if r < p:
            out.extend([c, c])  # duplicate
            i += 1
            continue
        if c.isalpha() and rng.random() < 0.05:
            c = c.upper() if c.islower() else c.lower()
        out.append(c)
        i += 1
    return "".join(out)


def mix(text_a, text_b, rng, frac_a=0.7):
    """~frac_a of A's tokens followed by ~(1-frac_a) of B's tokens."""
    a = text_a.split() if has_spaces(text_a) else list(text_a)
    b = text_b.split() if has_spaces(text_b) else list(text_b)
    ka, kb = max(1, int(len(a) * frac_a)), max(1, int(len(b) * (1 - frac_a)))
    ja, jb = (" " if has_spaces(text_a) else ""), (" " if has_spaces(text_b) else "")
    return ja.join(a[:ka]) + " " + jb.join(b[:kb])


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--num_langs", type=int, default=120, help="0 = all languages")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--per_lang_test", type=int, default=300,
                    help="devtest sentences per language used for each robustness slice")
    ap.add_argument("--cache", default=".flores_cache")
    ap.add_argument("--must_have", type=int, default=1,
                    help="1 = always include the MUST_HAVE major languages first (default), 0 = pure round-robin")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    splits = load_flores(args.cache)
    all_langs = sorted(set(splits["dev"]) & set(splits["devtest"]))
    langs = pick_languages(all_langs, args.num_langs, rng, MUST_HAVE if args.must_have else ())
    print(f"Using {len(langs)} of {len(all_langs)} available languages")

    # ---- train: dev split ----
    train_rows = [(t, lang) for lang in langs for t in splits["dev"][lang]]
    rng.shuffle(train_rows)
    write_csv(os.path.join(args.out, "train.csv"), train_rows, ["text", "label"])
    print(f"train: {len(train_rows)} rows")

    # ---- test: devtest split + robustness slices ----
    per_lang = {lang: list(splits["devtest"][lang]) for lang in langs}
    script_of = {l: l.split("_")[-1] for l in langs}
    clean, short, noisy, code_mixed, mixed_script = [], [], [], [], []
    for lang in langs:
        sents = per_lang[lang]
        rng.shuffle(sents)
        clean.extend((t, lang, "clean") for t in sents)
        for t in sents[: args.per_lang_test]:
            short.append((make_short(t, rng), lang, "short"))
            noisy.append((make_noisy(t, rng), lang, "noisy"))
            other = rng.choice([l for l in langs if l != lang])
            code_mixed.append((mix(t, rng.choice(per_lang[other]), rng), lang, "code_mixed"))
            diff = [l for l in langs if script_of[l] != script_of[lang]]
            if diff:
                o2 = rng.choice(diff)
                mixed_script.append((mix(t, rng.choice(per_lang[o2]), rng), lang, "mixed_script"))

    hdr = ["text", "label", "slice"]
    for name, rows in [("clean", clean), ("short", short), ("noisy", noisy),
                       ("code_mixed", code_mixed), ("mixed_script", mixed_script)]:
        write_csv(os.path.join(args.out, f"test_{name}.csv"), rows, hdr)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(langs, f, indent=2)
    print("test slices:", len(clean), len(short), len(noisy), len(code_mixed), len(mixed_script))


if __name__ == "__main__":
    main()
