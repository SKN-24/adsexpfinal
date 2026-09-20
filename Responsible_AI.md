# Responsible AI Report - Weather Rainfall Prediction

## 1. Model summary
| Item | Detail |
|---|---|
| Task | Regression - predict rainfall amount from weather observations |
| Model | Gradient Boosting Regressor (Huber loss, 300 trees, depth 4, learning rate 0.05) |
| Inputs | 23 features: temperature, humidity, wind, pressure, cloud cover, calendar features, 3 h / 6 h lags and changes |
| Data | cleaned_weather - merged_panvel_weather.csv/ https://github.com/SKN-24/adsexpfinal/blob/main/cleaned_weather%20-%20merged_panvel_weather.csv / licence of your dataset - not started|
| Train / test rows | 14222 / 3556 (time-ordered split, 80/20) |
| Test MAE / RMSE / R2 | 0.553 / 1.428 / 0.344 |

## 2. Intended use
- **Intended:** educational demonstration of an ML pipeline; indicative short-term rainfall estimates.
- **Not intended:** flood warnings, evacuation, agriculture or insurance decisions, or any safety-critical use. Always defer to official forecasts (e.g. IMD).

## 3. Fairness checklist
Weather data holds no protected personal attributes, so fairness here means *consistent reliability across seasons and conditions*.
- [x] Performance checked by **month / season** (table below)
- [x] Performance checked on **heavy-rain events** (top 10% of wet hours)
- [x] **False-alarm rate** checked on dry hours
- [ ] Coverage of all target locations confirmed (data is from: e.g. Mumbai, 2018-2023)

**Error by month**

| Month | Rows | MAE | Mean actual | Mean predicted |
|---|---|---|---|---|
| 6 | 720 | 0.208 | 0.279 | 0.161 |
| 7 | 1012 | 0.825 | 1.349 | 1.047 |
| 8 | 744 | 0.559 | 0.858 | 0.704 |
| 9 | 720 | 0.721 | 0.844 | 0.525 |
| 10 | 360 | 0.128 | 0.079 | 0.147 |

**Findings**
- On the heaviest 10% of wet hours (actual >= 3.51) the model **under-predicts** rainfall by 4.209 on average.
- 1.3% of dry hours were predicted as "rain likely" (prediction >= 1).
- Accuracy varies across months (see table) - use extra caution in months with higher MAE.

## 4. Privacy
- [x] No personal data is used - only meteorological measurements.
- [x] Dashboard uploads are processed in memory and not stored.
- [x] No credentials or private data are committed to the repository.

## 5. Consent and data provenance
- [x] Source and licence: Name / link / licence of your dataset
- [x] No individual users / devices are identifiable, so individual consent does not apply. Public-data terms of use are followed.

## 6. Transparency and explainability
- SHAP waterfall and summary plots are available in the dashboard and in `figures/`.
- Top three drivers by mean |SHAP|: `Humidity`, `Hour_cos`, `Wind_Speed`.
- Feature engineering is documented in `app.py` (`build_features`) and the notebook.

## 7. Robustness, monitoring and drift
- Drift is measured with PSI and the Kolmogorov-Smirnov test (dashboard tab and `figures/drift_report.csv`).
- Train-vs-test check: **0 of 23** features show major drift (PSI >= 0.25).
- Retraining trigger: PSI > 0.25 on key features or a sustained rise in MAE.

## 8. Known limitations
- Trained on limited data (e.g. Mumbai, 2018-2023); may not generalise to other places or climates.
- Needs correct 3 h / 6 h lag readings; wrong lags reduce accuracy.
- Point predictions only - no uncertainty interval.
- Rare extreme rainfall is under-represented.

## 9. Accountability
- Author: Aditya Saji , Sarvesh Mhatre ,Sauravkrishna Nair(SKN-24)
- Version 1.0 - issues and feedback via GitHub Issues.
