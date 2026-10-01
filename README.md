# E-commerce Customer Intelligence

RFM segmentation, purchase-intent prediction, churn prediction, and an item-item
recommender built on the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
and [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
datasets.

## Structure

```
01_data_understanding      raw-data inspection only, no modification
02_data_cleaning           cancellations, invalid qty/price, duplicates -> retail_clean.csv
03_eda                     revenue concentration, skew, country distribution
04_feature_engineering     RFM + log-transformed Frequency/Monetary, AvgOrderValue, IsUK
05_customer_segmentation   KMeans (K=4, chosen over statistically-cleaner K=2 for actionability)
06_purchase_prediction     dataset 2, classification + imbalance handling
07_churn_prediction        dataset 1, 60-day chronological-cutoff churn label
08_recommendation_system   item-item CF + popularity-bias fix + leave-last-purchase-out eval
09_dashboard               Streamlit app over the above (dashboard/app.py)
```

Anomaly detection, association rules, NLP, and demand forecasting were scoped out
and are listed under Future Work below, not built.

## Key findings, stated plainly

- **UK dominates revenue** (~85%) — `Country` is only usable as `IsUK` outside the
  UK subset; 43-way one-hot would be mostly noise.
- **Customer value is extremely right-skewed** (Monetary: £2.95–£580,987, ~670x
  spread) — log-transformed before clustering, or K-means finds "whales vs.
  everyone" and nothing else.
- **Segmentation:** K=4 is a *worse* statistical fit than K=2 (silhouette 0.362 vs
  0.419) but is the actionable choice — K=2 collapses to "good vs bad customer."
  New/Low-Engagement vs. Regular is split on Frequency (1.8 vs 5.9 orders), not
  Recency — the two clusters' mean Recency (99.9 vs 102.9 days) is too close to
  be a reliable signal; an earlier version of this notebook keyed on Recency and
  mislabeled the split.
- **Purchase prediction:** no clean winner across four approaches — Gradient
  Boosting has the best ROC-AUC (0.936), Random Forest + SMOTE the more balanced
  precision/recall tradeoff (shipped model). `PageValues` dominates feature
  importance and is semi-circular (it's Google Analytics' own revenue-contribution
  score) — the model is partly rediscovering a near-target feature, not purely
  finding subtle behavioral signal.
- **Churn prediction:** 63.1% 60-day churn rate, consistent with the median
  customer ordering only 3 times across ~2 years. Logistic Regression has the
  higher ROC-AUC (0.793 vs 0.782); Random Forest wins on churn-class recall (0.86
  vs 0.71) at the cost of non-churn recall — a tradeoff, not an objective win,
  and the shipped model (RF) reflects a judgment call that catching more true
  churners is worth more than the precision it costs.
- **Recommender:** naive item-item CF was 41.5% dominated by a single best-seller
  across 200 sampled customers. Fixed with a popularity floor (≥5 buyers) + tuned
  penalty (α=0.1), cutting dominance to 28.5% and raising unique products
  recommended from ~180 to ~261. A **leave-last-purchase-out evaluation** (added
  after the fact — accuracy was never checked at first, which is a real gap, not
  a footnote) shows the shipped recommender beats a trivial top-sellers baseline
  (HitRate@5 0.205 vs 0.134) and that the popularity fix cost no accuracy (0.205
  vs 0.202 for the unfixed version) — but absolute hit rate (~20%) means "better
  than nothing," not "solved."

## Environment note: pandas StringDtype cross-version pickling

The models in this bundle were rebuilt in an environment running **pandas 3.0**,
which defaults to a new `StringDtype` for text columns (`future.infer_string=True`).
Anything fitted or computed there (e.g. `OneHotEncoder` categories, the
recommender's product-name lookup) gets pickled with that dtype by default —
and a different pandas install (e.g. a typical `pip install pandas` on Windows,
which currently resolves to pandas 2.x) **cannot unpickle it**, raising
`NotImplementedError: (<StringDtype(storage='python', na_value=nan)>, ...)`
on load. This is a real cross-version incompatibility, not a corrupted file.

Fix applied: `pd.set_option('future.infer_string', False)` is set before any
data loading in every notebook that produces a saved model (04, 05, 06, 07, 08),
forcing plain `object` dtype throughout — which pickles compatibly across all
pandas versions. All four `.pkl` files in this bundle were verified (recursively,
not just spot-checked) to contain no `StringDtype`-backed objects before shipping.

If you retrain any model yourself in an environment with pandas ≥2.something
that also defaults to the new string dtype, keep that `pd.set_option` line at
the top of the notebook, or you'll reproduce this exact error.

## Environment note

`imbalanced-learn` (SMOTE) could not be installed in the environment this was
rebuilt in (no network access). Notebook 06's SMOTE step uses a manual k-NN
interpolation substitute (`notebooks/smote_pipeline.py`) with the same
train-fold-only discipline as real SMOTE. Results are close but not identical to
`imblearn`'s SMOTE (precision/recall 0.66/0.67 here vs. 0.68/0.70 originally;
ROC-AUC matches at 0.922 both times) — same conclusion, different exact numbers
for that one row. If `imbalanced-learn` is available in your environment, swap
back to the original `imblearn.pipeline.Pipeline` + `SMOTE` for an exact match.

## Running the dashboard

```
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Expects `../data/processed/` and `../models/` populated by notebooks 04–08.

## Future work (scoped out, not built)

- Anomaly detection (fraud / unusual order patterns)
- Association rule mining (market-basket analysis)
- NLP (review or support-ticket text, if such data existed for this dataset)
- Demand forecasting (time-series, per-product or aggregate)
- A proper SMOTE run under `imbalanced-learn` once network/package access allows
- Recommender: try matrix factorization (ALS/SVD) as a comparison point against
  item-item CF, and evaluate at higher K to see if HitRate improves meaningfully
