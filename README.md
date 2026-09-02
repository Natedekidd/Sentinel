# Sentinel

Account takeover detection for Nigerian bank transfers.

Most accounts are protected by a secret (PIN, OTP) that can be phished or stolen. Sentinel adds a second check: how the account is actually behaving, scored in real time, with a graduated response instead of a blunt block. It covers both app and USSD/feature phone users, and includes a working degraded mode fallback for when the scoring service itself is unreachable.

## How it works

```
src/generate_data.py   → synthetic users, sessions, transactions, ground truth labels
src/train_model.py     → feature engineering, two Isolation Forest models, evaluation
backend/main.py        → FastAPI: replays real scored transactions live, serves the kill switch
frontend/index.html    → dashboard: live ledger, reason sentences, degraded mode, key terms panel
```

Two models: one uses keystroke and session signals for app users, one uses transaction metadata only, since USSD users have no app collecting behavioral signals. Cold start accounts (no personal history) fall back to a peer cohort baseline.

## Quickstart

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend/requirements.txt

python src/generate_data.py     # writes data/
python src/train_model.py       # writes models/model.joblib, reports/scored_transactions.csv

cd backend && uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser.

## Results (held-out test set)

- ROC-AUC: 0.865 blended, 0.961 app channel, 0.772 USSD channel
- `stolen_otp`: 100% caught, including cold start accounts with zero personal history
- `sim_swap`: 100% caught, including on USSD with no app signal at all
- `drip` (patient, escalating transfers): 32.4% caught overall, but recall climbs from 0% on the first small transfer to 100% by the fifth to eighth step, as the rolling weekly velocity feature accumulates enough signal
- Known gap: cold start `drip` detection is weak (0/2 in test): a brand new account being slowly drained is the hardest case for this system, and we're reporting that honestly rather than hiding it

Full methodology, false positive rates on benign edge cases (shared phones, legitimate SIM changes), and the offline/outage design are in the write-up.

## Design choices worth knowing

- Personal and cohort baselines are computed from observed transaction history, not from the synthetic data generator's hidden ground truth parameters, using those would be leakage.
- The degraded mode fallback is a deliberately blunt static rule, not a second model. It never issues a Hold, only Pass or Step up, so an outage adds friction rather than blocking anyone's money outright. Every fallback decision is logged for rescoring once the real model reconnects.
- The frontend replays real scored output from `reports/scored_transactions.csv` rather than fabricating a live inference call, so every number shown in the demo came from the actual trained model.
