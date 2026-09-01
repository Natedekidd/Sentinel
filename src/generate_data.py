"""
A1 — Synthetic data generator for account-takeover detection.
Implements /synthetic_data_spec.md. Deterministic given RNG_SEED.

Output: users.csv, sessions.csv, transactions.csv, labels.csv, GENERATION.md
Design note: each transaction occurs within its own session (1:1 mapping) —
a scoping simplification for a single-transfer-per-session banking flow,
documented here so it's not mistaken for an oversight.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Location-independent: resolves relative to this file, not the cwd, so it
# works whether you run `python generate_data.py` from src/ or `python
# src/generate_data.py` from the project root.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- CONFIG ---
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_USERS = 5000
APP_RATIO = 0.5
COLD_START_RATIO = 0.10
N_COLD_START = int(N_USERS * COLD_START_RATIO)
N_ESTABLISHED = N_USERS - N_COLD_START

# Fixed-volume benign stress sets (guaranteed present, not left to random rate)
N_SHARED_DEVICE_BENIGN = 150
N_SIM_CHANGE_BENIGN = 150

# Attack volumes — established pool (~2% of established users total)
N_STOLEN_OTP = 50
N_DRIP = 25
N_SIM_SWAP_ATTACK = 25

# Attack volumes — cold_start pool (proves cohort-fallback catches it too)
N_STOLEN_OTP_COLD = 5
N_DRIP_COLD = 3
N_SIM_SWAP_ATTACK_COLD = 3

STATIC_FALLBACK_THRESHOLD = 50_000  # NGN — degraded-mode rule threshold; drip stays under this
GEN_END_DATE = datetime(2026, 8, 30)
HOUR_MODES = [9, 10, 12, 14, 17, 19, 21]
HOUR_MODE_WEIGHTS = [0.12, 0.14, 0.16, 0.14, 0.16, 0.16, 0.12]

# ------------------------------------------------------------- HELPERS -----

def new_recipient_id():
    return f"R{rng.integers(100000, 999999)}"


def sample_recipient_count():
    # most users 2-6 regulars, long tail to ~15
    n = rng.lognormal(mean=1.15, sigma=0.55)
    return int(np.clip(round(n), 1, 15))


# ---------------------------------------------------------- USER TABLE -----

def gen_users():
    rows = []
    cold_flags = np.array([False] * N_ESTABLISHED + [True] * N_COLD_START)
    rng.shuffle(cold_flags)
    channels = rng.choice(["app", "ussd"], size=N_USERS, p=[APP_RATIO, 1 - APP_RATIO])

    for i in range(N_USERS):
        uid = f"U{100000 + i}"
        cold = bool(cold_flags[i])
        channel = channels[i]

        if cold:
            account_age_days = int(rng.integers(1, 14))
            history_status = "cold_start"
        else:
            account_age_days = int(rng.integers(60, 900))
            history_status = "established"

        cohort_age_band = (
            "new" if account_age_days < 30 else
            "1-3mo" if account_age_days < 90 else
            "established"
        )
        baseline_avg_amount = float(np.clip(rng.lognormal(mean=8.5, sigma=0.9), 500, 500_000))
        cohort_txn_band = (
            "small" if baseline_avg_amount < 5_000 else
            "medium" if baseline_avg_amount < 30_000 else
            "large"
        )
        baseline_hour_mode = int(rng.choice(HOUR_MODES, p=HOUR_MODE_WEIGHTS))
        n_recip = sample_recipient_count()
        baseline_recipients = [new_recipient_id() for _ in range(n_recip)]
        primary_device = f"D{100000 + i}"

        # per-user keystroke baseline (app channel only — null downstream for ussd)
        keystroke_mean_base = float(np.clip(rng.normal(220, 45), 90, 500))
        keystroke_std_base = float(np.clip(rng.normal(35, 10), 5, 100))

        rows.append(dict(
            user_id=uid, channel=channel, account_age_days=account_age_days,
            history_status=history_status, cohort_age_band=cohort_age_band,
            cohort_txn_band=cohort_txn_band, baseline_avg_amount=baseline_avg_amount,
            baseline_hour_mode=baseline_hour_mode, baseline_recipients=baseline_recipients,
            primary_device=primary_device, keystroke_mean_base=keystroke_mean_base,
            keystroke_std_base=keystroke_std_base,
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------- ACTIVITY GEN ------

def window_start(user_row):
    span = min(user_row.account_age_days, 90 if user_row.history_status == "established" else 14)
    return GEN_END_DATE - timedelta(days=span), span


def random_timestamp(start, span_days, hour_mode):
    day_offset = rng.uniform(0, span_days)
    hour_jitter = np.clip(rng.normal(hour_mode, 2.5), 0, 23)
    ts = start + timedelta(days=day_offset, hours=hour_jitter)
    return ts


def gen_normal_activity(user_row, n_sessions):
    """Baseline organic sessions/transactions for one user."""
    start, span = window_start(user_row)
    recs = []
    for _ in range(n_sessions):
        ts = random_timestamp(start, span, user_row.baseline_hour_mode)
        use_baseline_recipient = rng.random() < 0.85
        recipient = (rng.choice(user_row.baseline_recipients) if use_baseline_recipient
                     else new_recipient_id())
        amount = float(np.clip(rng.lognormal(
            mean=np.log(max(user_row.baseline_avg_amount, 100)), sigma=0.4), 100, 2_000_000))

        rec = dict(
            user_id=user_row.user_id, channel=user_row.channel, timestamp=ts,
            device_fingerprint_id=(user_row.primary_device if user_row.channel == "app" else np.nan),
            amount=amount,
            recipient_id=recipient, archetype="normal", is_fraud=False,
            is_benign_anomaly="none",
        )
        if user_row.channel == "app":
            rec["keystroke_mean_ms"] = float(np.clip(rng.normal(
                user_row.keystroke_mean_base, 15), 50, 800))
            rec["keystroke_std_ms"] = float(np.clip(rng.normal(
                user_row.keystroke_std_base, 5), 5, 150))
            rec["field_pasted"] = bool(rng.random() < 0.05)
        else:
            rec["keystroke_mean_ms"] = np.nan
            rec["keystroke_std_ms"] = np.nan
            rec["field_pasted"] = np.nan
        recs.append(rec)
    return recs


def overlay_shared_device(user_row, recs):
    """Family member using the same phone — same device, different behavior. Benign."""
    if not recs:
        return recs
    n_affected = max(1, int(len(recs) * rng.uniform(0.2, 0.4)))
    idxs = rng.choice(len(recs), size=n_affected, replace=False)
    alt_mean = float(np.clip(rng.normal(220, 45), 90, 500))
    alt_amount_scale = rng.uniform(0.3, 2.5)
    for i in idxs:
        r = recs[i]
        r["is_benign_anomaly"] = "shared_device"
        r["amount"] = float(np.clip(r["amount"] * alt_amount_scale, 100, 2_000_000))
        if user_row.channel == "app":
            r["keystroke_mean_ms"] = float(np.clip(rng.normal(alt_mean, 15), 50, 800))
    return recs


def overlay_sim_change_benign(user_row, recs):
    """Legit device/SIM change — subsequent behavior still matches personal baseline."""
    if len(recs) < 4:
        return recs
    recs_sorted = sorted(recs, key=lambda r: r["timestamp"])
    change_idx = rng.integers(1, len(recs_sorted) - 1)
    new_device = f"D{rng.integers(900000, 999999)}"
    for r in recs_sorted[change_idx:]:
        r["device_fingerprint_id"] = new_device
        r["is_benign_anomaly"] = "sim_change"
        # amounts/recipients intentionally left as-is (already baseline-consistent)
    return recs_sorted


def make_attack_stolen_otp(user_row):
    start, span = window_start(user_row)
    ts = start + timedelta(days=rng.uniform(span * 0.6, span))
    off_hour = int((user_row.baseline_hour_mode + rng.integers(6, 12)) % 24)
    ts = ts.replace(hour=off_hour)
    amount = float(user_row.baseline_avg_amount * rng.uniform(5, 15))
    rec = dict(
        user_id=user_row.user_id, channel=user_row.channel, timestamp=ts,
        device_fingerprint_id=(user_row.primary_device if user_row.channel == "app" else np.nan),
        amount=amount,
        recipient_id=new_recipient_id(), archetype="takeover_stolen_otp",
        is_fraud=True, is_benign_anomaly="none",
    )
    if user_row.channel == "app":
        rec["keystroke_mean_ms"] = float(user_row.keystroke_mean_base * rng.uniform(0.4, 0.7))
        rec["keystroke_std_ms"] = float(user_row.keystroke_std_base * rng.uniform(1.5, 2.5))
        rec["field_pasted"] = True
    else:
        rec["keystroke_mean_ms"] = np.nan
        rec["keystroke_std_ms"] = np.nan
        rec["field_pasted"] = np.nan
    return [rec]


def make_attack_drip(user_row, step_range=(4, 9)):
    start, span = window_start(user_row)
    window_start_day = rng.uniform(0, max(span - 7, 0.1))
    n_steps = int(rng.integers(step_range[0], step_range[1]))
    start_amount = user_row.baseline_avg_amount * rng.uniform(0.05, 0.15)
    growth = (STATIC_FALLBACK_THRESHOLD * 0.9 / max(start_amount, 1)) ** (1 / max(n_steps - 1, 1))
    recipient = new_recipient_id()
    recs = []
    for step in range(n_steps):
        amount = float(min(start_amount * (growth ** step), STATIC_FALLBACK_THRESHOLD * 0.9))
        day_offset = window_start_day + step * (7 / n_steps) + rng.uniform(0, 0.3)
        ts = start + timedelta(days=day_offset)
        rec = dict(
            user_id=user_row.user_id, channel=user_row.channel, timestamp=ts,
            device_fingerprint_id=(user_row.primary_device if user_row.channel == "app" else np.nan),
            amount=amount,
            recipient_id=recipient, archetype="takeover_drip",
            is_fraud=True, is_benign_anomaly="none",
        )
        if user_row.channel == "app":
            rec["keystroke_mean_ms"] = float(np.clip(rng.normal(user_row.keystroke_mean_base, 15), 50, 800))
            rec["keystroke_std_ms"] = float(np.clip(rng.normal(user_row.keystroke_std_base, 5), 5, 150))
            rec["field_pasted"] = bool(rng.random() < 0.3)
        else:
            rec["keystroke_mean_ms"] = np.nan
            rec["keystroke_std_ms"] = np.nan
            rec["field_pasted"] = np.nan
        recs.append(rec)
    return recs


def make_attack_sim_swap(user_row):
    start, span = window_start(user_row)
    swap_day = rng.uniform(span * 0.3, span * 0.9)
    swap_ts = start + timedelta(days=swap_day)
    new_device = f"D{rng.integers(900000, 999999)}"
    n_burst = int(rng.integers(1, 4))
    recs = []
    for k in range(n_burst):
        ts = swap_ts + timedelta(hours=rng.uniform(0.5, 23))
        amount = float(user_row.baseline_avg_amount * rng.uniform(4, 12))
        rec = dict(
            user_id=user_row.user_id, channel=user_row.channel, timestamp=ts,
            device_fingerprint_id=new_device, amount=amount,
            recipient_id=new_recipient_id(), archetype="takeover_simswap",
            is_fraud=True, is_benign_anomaly="none",
        )
        if user_row.channel == "app":
            rec["keystroke_mean_ms"] = float(user_row.keystroke_mean_base * rng.uniform(0.5, 1.6))
            rec["keystroke_std_ms"] = float(user_row.keystroke_std_base * rng.uniform(1.2, 2.0))
            rec["field_pasted"] = bool(rng.random() < 0.5)
        else:
            rec["keystroke_mean_ms"] = np.nan
            rec["keystroke_std_ms"] = np.nan
            rec["field_pasted"] = np.nan
        recs.append(rec)
    return recs


# ------------------------------------------------------------- ASSEMBLE ----

def assign_pools(all_ids, app_ids, specs):
    """Disjoint pool assignment across archetypes.
    specs: list of (name, n, requires_app). requires_app=True draws only from
    app-channel users (device_fingerprint_id is an app-only signal — null for
    USSD, so device-based archetypes: shared_device, sim_change, sim_swap
    can't be built for USSD users without inventing a signal that doesn't exist)."""
    app_pool = list(app_ids)
    rng.shuffle(app_pool)
    all_pool = list(all_ids)
    rng.shuffle(all_pool)
    used = set()
    out = {}
    ai = 0
    for name, n, requires_app in specs:
        if requires_app:
            chosen = []
            while len(chosen) < n and ai < len(app_pool):
                cand = app_pool[ai]; ai += 1
                if cand not in used:
                    chosen.append(cand); used.add(cand)
            out[name] = set(chosen)
    gi = 0
    for name, n, requires_app in specs:
        if not requires_app:
            chosen = []
            while len(chosen) < n and gi < len(all_pool):
                cand = all_pool[gi]; gi += 1
                if cand not in used:
                    chosen.append(cand); used.add(cand)
            out[name] = set(chosen)
    return out


def build_dataset():
    users_df = gen_users()
    users_df = users_df.set_index("user_id", drop=False)

    established_ids = users_df.loc[users_df.history_status == "established", "user_id"].tolist()
    established_app_ids = users_df.loc[
        (users_df.history_status == "established") & (users_df.channel == "app"), "user_id"].tolist()
    cold_ids = users_df.loc[users_df.history_status == "cold_start", "user_id"].tolist()
    cold_app_ids = users_df.loc[
        (users_df.history_status == "cold_start") & (users_df.channel == "app"), "user_id"].tolist()

    est_pools = assign_pools(established_ids, established_app_ids, specs=[
        ("shared_device", N_SHARED_DEVICE_BENIGN, True),
        ("sim_change", N_SIM_CHANGE_BENIGN, False),
        ("sim_swap", N_SIM_SWAP_ATTACK, False),
        ("stolen_otp", N_STOLEN_OTP, False),
        ("drip", N_DRIP, False),
    ])
    cold_pools = assign_pools(cold_ids, cold_app_ids, specs=[
        ("sim_swap", N_SIM_SWAP_ATTACK_COLD, False),
        ("stolen_otp", N_STOLEN_OTP_COLD, False),
        ("drip", N_DRIP_COLD, False),
    ])

    all_recs = []
    for uid, row in users_df.iterrows():
        is_cold = row.history_status == "cold_start"

        if not is_cold:
            n_sessions = int(rng.integers(20, 60))
            recs = gen_normal_activity(row, n_sessions)

            if uid in est_pools["shared_device"]:
                recs = overlay_shared_device(row, recs)
            if uid in est_pools["sim_change"]:
                recs = overlay_sim_change_benign(row, recs)
            if uid in est_pools["stolen_otp"]:
                recs += make_attack_stolen_otp(row)
            if uid in est_pools["drip"]:
                recs += make_attack_drip(row)
            if uid in est_pools["sim_swap"]:
                recs += make_attack_sim_swap(row)
        else:
            # Cold-start: generate attack records first, then budget normal
            # sessions so total stays <5 — preserves the history_status
            # invariant even when an attack is overlaid (needed to genuinely
            # test the cohort-fallback path with zero personal baseline).
            attack_recs = []
            if uid in cold_pools["stolen_otp"]:
                attack_recs += make_attack_stolen_otp(row)
            if uid in cold_pools["drip"]:
                attack_recs += make_attack_drip(row, step_range=(2, 4))
            if uid in cold_pools["sim_swap"]:
                attack_recs += make_attack_sim_swap(row)

            max_normal = max(0, 4 - len(attack_recs))
            n_sessions = int(rng.integers(0, max_normal + 1))
            recs = gen_normal_activity(row, n_sessions) if n_sessions else []
            recs += attack_recs

        all_recs.extend(recs)

    tx_df = pd.DataFrame(all_recs)
    tx_df = tx_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ---- derive recipient_is_new and is_new_device chronologically, per user ----
    recipient_is_new, is_new_device = [], []
    seen_recipients, seen_devices = {}, {}
    baseline_recip_map = users_df["baseline_recipients"].to_dict()

    for _, r in tx_df.iterrows():
        uid = r.user_id
        seen_recip = seen_recipients.setdefault(uid, set(baseline_recip_map[uid]))
        seen_dev = seen_devices.setdefault(uid, set())

        recipient_is_new.append(r.recipient_id not in seen_recip)
        seen_recip.add(r.recipient_id)

        is_new_device.append(r.device_fingerprint_id not in seen_dev)
        seen_dev.add(r.device_fingerprint_id)

    tx_df["recipient_is_new"] = recipient_is_new
    tx_df["is_new_device"] = is_new_device
    tx_df["hour_of_day"] = tx_df["timestamp"].dt.hour
    tx_df["day_of_week"] = tx_df["timestamp"].dt.dayofweek

    # ---- ids ----
    tx_df["transaction_id"] = [f"T{100000+i}" for i in range(len(tx_df))]
    tx_df["session_id"] = [f"S{100000+i}" for i in range(len(tx_df))]

    # ---- split into output tables ----
    sessions_df = tx_df[[
        "session_id", "user_id", "channel", "timestamp", "device_fingerprint_id",
        "keystroke_mean_ms", "keystroke_std_ms", "field_pasted", "is_new_device",
    ]].rename(columns={"timestamp": "start_time"})

    transactions_df = tx_df[[
        "transaction_id", "session_id", "user_id", "timestamp", "amount",
        "recipient_id", "recipient_is_new", "hour_of_day", "day_of_week",
    ]]

    labels_df = tx_df[[
        "user_id", "session_id", "transaction_id", "archetype", "is_fraud", "is_benign_anomaly",
    ]]

    users_out = users_df.drop(columns=["primary_device", "keystroke_mean_base",
                                        "keystroke_std_base", "baseline_recipients"]).copy()
    users_out["baseline_recipient_count"] = users_df["baseline_recipients"].apply(len)

    return users_out.reset_index(drop=True), sessions_df, transactions_df, labels_df


if __name__ == "__main__":
    users_out, sessions_df, transactions_df, labels_df = build_dataset()

    users_out.to_csv(DATA_DIR / "users.csv", index=False)
    sessions_df.to_csv(DATA_DIR / "sessions.csv", index=False)
    transactions_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    labels_df.to_csv(DATA_DIR / "labels.csv", index=False)

    with open(DATA_DIR / "GENERATION.md", "w") as f:
        f.write(f"""# Generation notes

- Seed: {RNG_SEED} (numpy default_rng) — fully reproducible
- N_USERS={N_USERS}, cold_start={N_COLD_START} ({COLD_START_RATIO:.0%}), established={N_ESTABLISHED}
- Channel split: app/ussd = {APP_RATIO:.0%}/{1-APP_RATIO:.0%}
- Benign stress sets (fixed volume): shared_device={N_SHARED_DEVICE_BENIGN}, sim_change={N_SIM_CHANGE_BENIGN}
- Attacks (established): stolen_otp={N_STOLEN_OTP}, drip={N_DRIP}, sim_swap={N_SIM_SWAP_ATTACK}
- Attacks (cold_start overlay): stolen_otp={N_STOLEN_OTP_COLD}, drip={N_DRIP_COLD}, sim_swap={N_SIM_SWAP_ATTACK_COLD}
- Static fallback threshold used to cap drip escalation: {STATIC_FALLBACK_THRESHOLD:,} NGN
- Amounts: lognormal per-user baseline, session amounts lognormal around that baseline
- recipient_is_new / is_new_device: derived chronologically per user, not randomized independently
- Design simplification: 1 transaction per session (documented, not an oversight)
""")

    # ---- sanity checks (per spec) ----
    print("Row counts:", len(users_out), len(sessions_df), len(transactions_df), len(labels_df))
    print("\nArchetype counts:\n", labels_df.archetype.value_counts())
    print("\nBenign anomaly counts:\n", labels_df.is_benign_anomaly.value_counts())
    print("\nFraud rate: {:.3%}".format(labels_df.is_fraud.mean()))
    drip_max = transactions_df.merge(labels_df, on="transaction_id")
    drip_max = drip_max[drip_max.archetype == "takeover_drip"].amount.max()
    print(f"\nMax single drip transaction amount: {drip_max:,.0f} (threshold {STATIC_FALLBACK_THRESHOLD:,})")
