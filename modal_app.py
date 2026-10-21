"""
Deploy the Polyglot-120 API on Modal (free Starter plan, scales to zero).

One-time setup:
    pip install modal
    modal setup                     # opens the browser, creates your token

Deploy (run from the project root, the folder that contains app/):
    modal deploy modal_app.py       # prints the public URL when done

The image bakes in the fine-tuned model and the fastText fallback, so a cold
start only loads weights from disk instead of downloading 1.1 GB each time.
Rebuild (pick up a new model version) with:  modal deploy modal_app.py --force-build
"""
import modal

HF_MODEL = "srikar1503/polyglot-120"   # Hugging Face model repo to serve
MODEL_VERSION = "2"                    # bump after re-uploading a new model so the image re-downloads it
CONF_THRESHOLD = "0.8"                 # chosen from the routing table in reports/REPORT.md

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    # CPU-only torch from the PyTorch index (the PyPI build pulls ~2.5 GB of CUDA libs)
    .pip_install("torch>=2.2", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers>=4.44", "sentencepiece>=0.2.0", "protobuf>=4.25", "numpy",
        "fastapi>=0.112", "uvicorn[standard]>=0.30", "pydantic>=2.8",
        "fasttext-wheel>=0.9.2", "huggingface_hub>=0.24",
    )
    .run_commands(
        # fastText lid.176 fallback (~900 KB, 176 languages)
        "python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', '/root/lid.176.ftz')\"",
        # bake the fine-tuned model into the image
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{HF_MODEL}', "
        f"local_dir='/root/model', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])\"  # v{MODEL_VERSION}",
    )
    .env({
        "MODEL_ID": "/root/model",
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "FASTTEXT_PATH": "/root/lid.176.ftz",
        "TORCH_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir("app", remote_path="/root/app")   # the FastAPI service
)

app = modal.App("polyglot-120")   # `modal deploy modal_app.py` looks for a variable named `app`
modal_app = app                    # alias, so `modal deploy modal_app.py::modal_app` works too


# scaledown_window: seconds a container stays alive after its last request. Idle time is
# billed, so keep it short on the free plan (5 min) and raise it if you have credits to spare.
@app.function(image=image, cpu=2.0, memory=4096, scaledown_window=300, timeout=120)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(label="polyglot-120")
def api():
    # Runs once per container start: imports app/main.py, which loads the model.
    from app.main import app as fastapi_app
    return fastapi_app
