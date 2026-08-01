# Contactless Fingerprint Quality Assessment & Scoring Pipeline

Assignment 4 -- YellowSense Technologies technical assessment.

An image quality-control gate for contactless (phone-camera) fingerprint
capture. Runs 5 cheap checks (blur, brightness, glare, ROI completeness,
ridge clarity), combines them into a 0-100 composite score, and returns
pass/fail plus a plain-language retake instruction.

## Project structure

```
contactless-fingerprint-qc/
├── quality_assessment.py   # 5 metric functions + quality_gate()
├── quality_app.py          # Streamlit dashboard
├── test_quality.py         # Batch test over test_dataset/
├── requirements.txt
├── README.md
└── test_dataset/
    ├── good/     (5 images)
    ├── blurry/   (5 images)
    ├── dark/     (5 images)
    └── glare/    (5 images)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Streamlit dashboard

```bash
streamlit run quality_app.py
```

Upload a fingerprint photo, adjust thresholds in the sidebar, and watch the
composite score and pass/fail badges update.

## Run the batch test

1. Drop your own captured photos into `test_dataset/good/`,
   `test_dataset/blurry/`, `test_dataset/dark/`, `test_dataset/glare/`
   (5 each).
2. Run:

```bash
python test_quality.py
```

This prints a summary table and writes `test_results.csv`.

## Notes on the metrics

- **Blur** -- Laplacian variance (edge sharpness).
- **Brightness** -- mean grayscale intensity, ideal target ~128.
- **Glare** -- fraction of near-saturated pixels (I > 240).
- **ROI completeness** -- Otsu-thresholded foreground area / total area
  (a quick stand-in for real segmentation).
- **Ridge clarity** -- variance of a single-orientation Gabor filter
  response; a simplification of the multi-orientation Gabor bank a
  production system would use.

Composite score = weighted sum of the 5 normalized metrics
(weights: blur 0.25, brightness 0.15, glare 0.15, ROI 0.20, ridge 0.25).
Pass requires composite >= 60 **and** no individual hard failure.
