"""
A1 — Anomaly model: feature engineering, training, scoring, reason codes,
graduated response tiers, and evaluation against ground truth.

Design choices worth knowing before reading the code:
- Personal baselines (amount, typical hour) are computed EMPIRICALLY from each
  user's own prior observed transactions (expanding window, shifted by 1 —
  causal, no look-ahead). We deliberately do NOT read users.csv's
  baseline_avg_amount / baseline_hour_mode columns as model features — those
  are hidden generative ground truth a real bank would never have. Using them
  would be leakage dressed up as a baseline.
- Cold-start fallback uses a cohort baseline (mean over established users in
  the same cohort_age_band x cohort_txn_band), computed only from the
  training split's normal-labeled rows — again no ground-truth shortcuts.
- Two models: iso_app (keystroke + metadata features, app channel only) and
  iso_meta (metadata-only features, ALL channels — this is what actually
  protects USSD users, and is also the honest fallback if the app model is
  unavailable).
- Both trained only on is_fraud == False rows in the TRAIN split — "learn
  normal, flag deviation," per the spec.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
from pathlib import Path

RNG_SEED = 42

# Location-independent, matches the Sentinel/ project layout:
#   src/train_model.py reads from ../data, writes to ../models and ../reports
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- LOAD ------

users = pd.read_csv(DATA_DIR / "users.csv")
sessions = pd.read_csv(DATA_DIR / "sessions.csv", parse_dates=["start_time"])
transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
labels = pd.read_csv(DATA_DIR / "labels.csv")

df = transactions.merge(
    sessions[["session_id", "channel", "device_fingerprint_id", "keystroke_mean_ms",
              "keystroke_std_ms", "field_pasted", "is_new_device"]], on="session_id"
).merge(
    users[["user_id", "history_status", "cohort_age_band", "cohort_txn_band"]], on="user_id"
).merge(
    labels[["transaction_id", "archetype", "is_fraud", "is_benign_anomaly"]], on="transaction_id"
)
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# ------------------------------------------------------ TRAIN/TEST SPLIT ---
# Split by user_id so no user's history leaks across the split. Stratify on
# a composite key (history_status x has_any_fraud) so both splits get a fair
# share of each attack archetype and of cold-start users.
user_flags = df.groupby("user_id").agg(
    history_status=("history_status", "first"),
    has_fraud=("is_fraud", "max"),
).reset_index()
user_flags["strata"] = user_flags.history_status + "_" + user_flags.has_fraud.astype(str)

train_users, test_users = train_test_split(
    user_flags, test_size=0.2, random_state=RNG_SEED, stratify=user_flags["strata"]
)
train_ids, test_ids = set(train_users.user_id), set(test_users.user_id)

# Cold-start attack pools are tiny (3 users each for drip / sim_swap), so a
# random split can place ALL of a rare pool into train by chance — which
# would silently make "cohort fallback catches it" untestable for that
# archetype, not proven true. Force at least one cold-start user per
# (archetype, cold_start) combination into test, deterministically, so every
# archetype the generator overlaid onto cold-start users actually gets
# evaluated. This only moves users FROM train TO test (never removes test
# coverage), and only touches pools that would otherwise have zero test
# representation.
cold_attack_users = df[(df.history_status == "cold_start") & (df.is_fraud)].groupby(
    "archetype")["user_id"].unique()
for arch, uids in cold_attack_users.items():
    uids_sorted = sorted(uids)
    if not any(u in test_ids for u in uids_sorted):
        forced = uids_sorted[0]
        train_ids.discard(forced)
        test_ids.add(forced)
        print(f"[split] forced cold-start '{arch}' user {forced} into test "
              f"(pool of {len(uids_sorted)} had zero test representation)")

df["split"] = np.where(df.user_id.isin(train_ids), "train", "test")

# ------------------------------------------------------- FEATURE ENGINEERING

# --- personal expanding baselines (causal: only prior rows, via shift(1)) ---
# NOTE: this expanding window uses ALL of a user's prior transactions,
# including any that were themselves fraud (e.g. earlier drip steps).
# That's intentional, not an oversight: a live system has no ground-truth
# label for a user's own past transactions at scoring time, so it can never
# selectively exclude "known fraud" from a personal baseline in production
# either. Excluding it here would be an oracle shortcut the deployed system
# won't have. (Cohort fallback below is different — that's a population
# baseline computed offline from labeled training data, which a bank could
# reasonably audit and curate.)
df["n_prior"] = df.groupby("user_id").cumcount()

df["amt_sum_prior"] = df.groupby("user_id")["amount"].transform(lambda s: s.expanding().sum().shift(1))
df["personal_amt_baseline"] = df["amt_sum_prior"] / df["n_prior"].replace(0, np.nan)

df["hour_sin"] = np.sin(2 * np.pi * df.hour_of_day / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.hour_of_day / 24)
sin_sum = df.groupby("user_id")["hour_sin"].transform(lambda s: s.expanding().sum().shift(1))
cos_sum = df.groupby("user_id")["hour_cos"].transform(lambda s: s.expanding().sum().shift(1))
mean_sin = sin_sum / df["n_prior"].replace(0, np.nan)
mean_cos = cos_sum / df["n_prior"].replace(0, np.nan)
df["personal_hour_mode"] = (np.arctan2(mean_sin, mean_cos) / (2 * np.pi) * 24) % 24

ks_sum = df.groupby("user_id")["keystroke_mean_ms"].transform(lambda s: s.expanding().sum().shift(1))
df["personal_ks_mean"] = ks_sum / df["n_prior"].replace(0, np.nan)
df["personal_ks_std"] = df.groupby("user_id")["keystroke_mean_ms"].transform(
    lambda s: s.expanding().std().shift(1))

# --- cohort fallback baselines (from TRAIN split, normal rows only) --------
cohort_src = df[(df.split == "train") & (df.is_fraud == False)]  # noqa: E712
cohort_amt = cohort_src.groupby(["cohort_age_band", "cohort_txn_band"])["amount"].mean()
cohort_hour_sin = cohort_src.groupby(["cohort_age_band", "cohort_txn_band"])["hour_sin"].mean()
cohort_hour_cos = cohort_src.groupby(["cohort_age_band", "cohort_txn_band"])["hour_cos"].mean()
cohort_ks_mean = cohort_src[cohort_src.channel == "app"].groupby(
    ["cohort_age_band", "cohort_txn_band"])["keystroke_mean_ms"].mean()
cohort_ks_std = cohort_src[cohort_src.channel == "app"].groupby(
    ["cohort_age_band", "cohort_txn_band"])["keystroke_mean_ms"].std()

cohort_key = list(zip(df.cohort_age_band, df.cohort_txn_band))
df["cohort_amt_fb"] = [cohort_amt.get(k, cohort_amt.mean()) for k in cohort_key]
df["cohort_hour_fb"] = [
    (np.arctan2(cohort_hour_sin.get(k, 0), cohort_hour_cos.get(k, 1)) / (2 * np.pi) * 24) % 24
    for k in cohort_key
]
df["cohort_ks_mean_fb"] = [cohort_ks_mean.get(k, cohort_ks_mean.mean()) for k in cohort_key]
df["cohort_ks_std_fb"] = [cohort_ks_std.get(k, cohort_ks_std.mean()) for k in cohort_key]

# --- fill personal baselines from cohort where no prior history exists -----
df["amt_baseline"] = df["personal_amt_baseline"].fillna(df["cohort_amt_fb"])
df["hour_baseline"] = df["personal_hour_mode"].fillna(df["cohort_hour_fb"])
df["ks_mean_baseline"] = df["personal_ks_mean"].fillna(df["cohort_ks_mean_fb"])
df["ks_std_baseline"] = df["personal_ks_std"].fillna(df["cohort_ks_std_fb"]).fillna(30.0).clip(lower=5)

# --- final features ---------------------------------------------------------
df["amount_ratio"] = df["amount"] / df["amt_baseline"].clip(lower=100)
hd = (df.hour_of_day - df.hour_baseline).abs()
df["hour_deviation"] = np.minimum(hd, 24 - hd)

# trailing velocity windows (time-based rolling, per user, causal-inclusive)
df = df.sort_values(["user_id", "timestamp"])
roll = df.set_index("timestamp").groupby("user_id")["amount"]
df["trailing_24h_count"] = roll.rolling("24h").count().reset_index(level=0, drop=True).values
df["trailing_24h_sum"] = roll.rolling("24h").sum().reset_index(level=0, drop=True).values
df["trailing_7d_count"] = roll.rolling("7D").count().reset_index(level=0, drop=True).values
df["trailing_7d_sum"] = roll.rolling("7D").sum().reset_index(level=0, drop=True).values
df["trailing_7d_sum_ratio"] = df["trailing_7d_sum"] / df["amt_baseline"].clip(lower=100)

df["recipient_is_new"] = df["recipient_is_new"].astype(int)
df["is_new_device"] = df["is_new_device"].astype(int)

df["keystroke_deviation"] = (
    (df["keystroke_mean_ms"] - df["ks_mean_baseline"]).abs() / df["ks_std_baseline"]
)
df["field_pasted_flag"] = df["field_pasted"].fillna(False).astype(int)

META_FEATURES = [
    "amount_ratio", "hour_deviation", "recipient_is_new", "is_new_device",
    "trailing_24h_count", "trailing_7d_count", "trailing_7d_sum_ratio",
]
APP_FEATURES = META_FEATURES + ["keystroke_deviation", "field_pasted_flag"]

df[META_FEATURES + ["keystroke_deviation", "field_pasted_flag"]] = (
    df[META_FEATURES + ["keystroke_deviation", "field_pasted_flag"]].fillna(0)
)

# ------------------------------------------------------------- TRAIN -------

train_normal = df[(df.split == "train") & (df.is_fraud == False)]  # noqa: E712

iso_meta = IsolationForest(n_estimators=200, contamination=0.02, random_state=RNG_SEED)
iso_meta.fit(train_normal[META_FEATURES])

train_normal_app = train_normal[train_normal.channel == "app"]
iso_app = IsolationForest(n_estimators=200, contamination=0.02, random_state=RNG_SEED)
iso_app.fit(train_normal_app[APP_FEATURES])

# ------------------------------------------------------------- SCORE -------
# sklearn score_samples: higher = more normal. Flip sign so higher = riskier.

df["meta_risk"] = -iso_meta.score_samples(df[META_FEATURES])
app_mask = df.channel == "app"
df.loc[app_mask, "app_risk"] = -iso_app.score_samples(df.loc[app_mask, APP_FEATURES])

# primary_risk: app model for app users (falls back to meta if unavailable), meta for ussd
df["primary_risk"] = np.where(df.channel == "app", df["app_risk"], df["meta_risk"])

# ------------------------------------------------- GRADUATED RESPONSE TIERS
# Thresholds calibrated on the TRAIN model's own normal training scores —
# not on the test set, and not using any fraud labels — so this is something
# a live system could actually compute.
train_normal_meta_scores = -iso_meta.score_samples(train_normal[META_FEATURES])
train_normal_app_scores = -iso_app.score_samples(train_normal_app[APP_FEATURES])

meta_hold_thr, meta_stepup_thr = np.percentile(train_normal_meta_scores, [99.5, 97.5])
app_hold_thr, app_stepup_thr = np.percentile(train_normal_app_scores, [99.5, 97.5])


def tier_row(row):
    thr_hold, thr_stepup = (app_hold_thr, app_stepup_thr) if row.channel == "app" else (meta_hold_thr, meta_stepup_thr)
    if row.primary_risk >= thr_hold:
        return "HOLD"
    if row.primary_risk >= thr_stepup:
        return "STEP_UP"
    return "PASS"


df["tier"] = df.apply(tier_row, axis=1)

# --------------------------------------------------------------- REASONS ---

REASON_TEMPLATES = [
    ("amount_ratio", lambda r: r.amount_ratio > 2.5,
     lambda r: f"amount (₦{r.amount:,.0f}) is {r.amount_ratio:.1f}x this account's typical transfer"),
    ("trailing_7d_sum_ratio", lambda r: r.trailing_7d_sum_ratio > 2.5,
     lambda r: f"total transfers this week are {r.trailing_7d_sum_ratio:.1f}x the usual weekly total"),
    ("is_new_device", lambda r: r.is_new_device == 1,
     lambda r: "sent from a device or line not seen on this account before"),
    ("recipient_is_new", lambda r: r.recipient_is_new == 1,
     lambda r: "sent to a recipient this account hasn't paid before"),
    ("hour_deviation", lambda r: r.hour_deviation > 6,
     lambda r: "sent at a time of day unusual for this account"),
    ("keystroke_deviation", lambda r: r.channel == "app" and r.keystroke_deviation > 3,
     lambda r: "typing pattern differs from this account's usual pattern"),
    ("field_pasted_flag", lambda r: r.channel == "app" and r.field_pasted_flag == 1,
     lambda r: "PIN/OTP field was pasted rather than typed"),
]

STRENGTH = {
    "amount_ratio": lambda r: max(r.amount_ratio - 1, 0),
    "trailing_7d_sum_ratio": lambda r: max(r.trailing_7d_sum_ratio - 1, 0),
    "is_new_device": lambda r: 3.0 if r.is_new_device == 1 else 0,
    "recipient_is_new": lambda r: 1.5 if r.recipient_is_new == 1 else 0,
    "hour_deviation": lambda r: r.hour_deviation / 6,
    "keystroke_deviation": lambda r: r.keystroke_deviation,
    "field_pasted_flag": lambda r: 1.0 if r.field_pasted_flag == 1 else 0,
}


def reason_sentence(row):
    if row.tier == "PASS":
        return ""
    triggered = [(name, STRENGTH[name](row)) for name, cond, _ in REASON_TEMPLATES if cond(row)]
    triggered.sort(key=lambda t: -t[1])
    top = triggered[:2]
    if not top:
        return "Overall pattern on this account looks unusual compared to its normal history."
    parts = [msg(row) for name, _, msg in REASON_TEMPLATES if name in {n for n, _ in top}]
    return "Flagged because: " + "; and ".join(parts) + "."


df["reason"] = df.apply(reason_sentence, axis=1)

# ------------------------------------------------------------ EVALUATION ---

test = df[df.split == "test"].copy()
test["flagged"] = test.tier.isin(["STEP_UP", "HOLD"])

print("=" * 70)
print("TEST SET SUMMARY")
print("=" * 70)
print(f"Rows: {len(test):,} | Users: {test.user_id.nunique():,} | "
      f"Fraud rows: {test.is_fraud.sum()} | Fraud users: {test[test.is_fraud].user_id.nunique()}")

auc_all = roc_auc_score(test.is_fraud, test.primary_risk)
ap_all = average_precision_score(test.is_fraud, test.primary_risk)
print(f"\nROC-AUC (primary_risk vs is_fraud): {auc_all:.3f}")
print(f"Average precision: {ap_all:.3f}")

for ch in ["app", "ussd"]:
    sub = test[test.channel == ch]
    if sub.is_fraud.sum() > 0:
        auc = roc_auc_score(sub.is_fraud, sub.primary_risk)
        print(f"  {ch}: ROC-AUC={auc:.3f}  (n={len(sub)}, fraud={sub.is_fraud.sum()})")

print("\n--- Confusion at tier level (flagged = STEP_UP or HOLD) ---")
tp = ((test.flagged) & (test.is_fraud)).sum()
fn = ((~test.flagged) & (test.is_fraud)).sum()
fp = ((test.flagged) & (~test.is_fraud)).sum()
tn = ((~test.flagged) & (~test.is_fraud)).sum()
precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall = tp / (tp + fn) if (tp + fn) else float("nan")
print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
print(f"Precision={precision:.3f}  Recall={recall:.3f}")

print("\n--- Recall by archetype ---")
for arch in ["takeover_stolen_otp", "takeover_drip", "takeover_simswap"]:
    sub = test[test.archetype == arch]
    if len(sub):
        r = sub.flagged.mean()
        print(f"  {arch}: {r:.1%} caught ({sub.flagged.sum()}/{len(sub)} rows)")

print("\n--- Recall by archetype x history_status (proves/disproves cohort fallback) ---")
for arch in ["takeover_stolen_otp", "takeover_drip", "takeover_simswap"]:
    for hist in ["established", "cold_start"]:
        sub = test[(test.archetype == arch) & (test.history_status == hist)]
        if len(sub):
            print(f"  {arch} / {hist}: {sub.flagged.mean():.1%} ({sub.flagged.sum()}/{len(sub)})")
        else:
            print(f"  {arch} / {hist}: NO TEST ROWS — cannot evaluate")

print("\n--- Drip: recall by escalation step (does baseline drift suppress later steps?) ---")
drip = test[test.archetype == "takeover_drip"].sort_values(["user_id", "timestamp"]).copy()
drip["step"] = drip.groupby("user_id").cumcount() + 1
print(drip.groupby("step").flagged.agg(["mean", "count"]))

print("\n--- False-positive rate on BENIGN stress sets (must stay low) ---")
for anomaly in ["shared_device", "sim_change"]:
    sub = test[test.is_benign_anomaly == anomaly]
    if len(sub):
        print(f"  {anomaly}: {sub.flagged.mean():.1%} flagged ({sub.flagged.sum()}/{len(sub)} rows) — false positives")

print("\n--- USSD sim-swap specifically (the case this whole extension was for) ---")
sub = test[(test.archetype == "takeover_simswap") & (test.channel == "ussd")]
if len(sub):
    print(f"  {sub.flagged.sum()}/{len(sub)} caught ({sub.flagged.mean():.1%})")
else:
    print("  (none in this test split)")

# --------------------------------------------------------------- SAVE ------

joblib.dump({"iso_meta": iso_meta, "iso_app": iso_app,
             "meta_features": META_FEATURES, "app_features": APP_FEATURES,
             "thresholds": {"meta_hold": meta_hold_thr, "meta_stepup": meta_stepup_thr,
                             "app_hold": app_hold_thr, "app_stepup": app_stepup_thr}},
            MODELS_DIR / "model.joblib")

out_cols = ["transaction_id", "user_id", "channel", "timestamp", "amount", "history_status",
            "split", "archetype", "is_fraud", "is_benign_anomaly", "primary_risk", "tier", "reason"]
df[out_cols].to_csv(REPORTS_DIR / "scored_transactions.csv", index=False)

print(f"\nSaved {MODELS_DIR/'model.joblib'} and {REPORTS_DIR/'scored_transactions.csv'}")
