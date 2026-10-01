# YouTube Comment Sentiment

Manifest V3 Chrome extension and a decoupled FastAPI service for analysing YouTube comments in batches. The extension collects up to 1,000 rendered comments, sends them to the API in resilient chunks, and displays overall polarity, calibrated class probabilities, and a per-comment sentiment spectrum.

## Quick start

1. Create a Python environment and install the service:
   ```powershell
   cd api
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   $env:MODEL_PATH = "./model" # optional; see training below
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
2. In Chrome, open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**, then select the `extension` directory.
3. Open a YouTube video, scroll until comments load, open the extension popup, enter the API URL if needed, and select **Analyse comments**.

The API runs a deterministic lexicon fallback when no local transformer model is configured, so the extension remains usable in a fresh clone. For production, export the fine-tuned model to `api/model` or set `MODEL_PATH`.

## API

`POST /v1/sentiment/analyze`

```json
{"comments":[{"id":"abc","text":"Great video!"}]}
```

The response contains one result for every accepted comment (`negative`, `neutral`, `positive` probabilities), aggregate counts, and mean probability distribution. The maximum batch is 2,000 comments, with a per-comment length limit of 2,000 characters.

`GET /health` reports whether a transformer or fallback analyser is active.

## Fine-tuning and evaluation

Place a CSV with `text` and `label` columns in `training/data/train.csv` and `training/data/test.csv`; labels must be `negative`, `neutral`, or `positive`. Then run:

```powershell
cd training
pip install -r requirements.txt
python train.py --train data/train.csv --test data/test.csv --output ../api/model
```

The script fine-tunes `distilbert-base-uncased`, writes a held-out `metrics.json`, and fails the process if accuracy is below `0.894` or ROC-AUC is below `0.91`. Those thresholds are quality gates—not fabricated results—and must be validated with your real, representative held-out data.

## Notes

- The extension uses local, dependency-free SVG/canvas rendering rather than remote chart CDNs: Manifest V3 CSP blocks remotely hosted scripts. The visualisation presents the same distributions/spectrum normally implemented with Chart.js/D3, without making users trust a third party script.
- YouTube markup changes regularly. The collector targets current comment-renderer elements, de-duplicates by stable comment ID when available, and stops after a bounded scrolling cycle.
- This is sentiment classification, not moderation. Do not use it alone for enforcement decisions.
