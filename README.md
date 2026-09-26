# Adverse-action SHAP demo (offline computation)

Offline machine-learning prep behind the
[Adverse-Action Notice Generator](https://movahedi.ca/tools/adverse-action/)
demo on movahedi.ca.

## What it does

`shap_prep.py` downloads the UCI Statlog German Credit dataset (1,000 rows),
trains a small logistic regression (C=1.0, standardized features, 800/200
stratified split, random_state=42, test accuracy 0.72), picks four diverse
sample applicants from the test set (two denied, two approved), and computes
SHAP values with `shap.LinearExplainer`. The results are written to
`shap_adverse_action.json`, which the website embeds directly so the demo
page runs with no backend and no visitor data leaving the browser.

## Run it

```bash
pip install -r requirements.txt
python shap_prep.py
```

## Honest notes

- German Credit stores ordered categorical codes. They are mapped to
  human-meaningful notice features; revolving utilization, late payments, DTI,
  and income are deterministic seeded proxies derived from the real codes
  (see `map_row` and the dataset note in the JSON). Treat magnitudes as
  illustrative.
- The demo model is small, old-data, and educational. It is not a lending
  model and its attributions are not legal findings.
- One approved sample's top positive driver is age, an artifact the data
  taught the model. The demo page flags this explicitly: age must not drive
  real credit decisions.
