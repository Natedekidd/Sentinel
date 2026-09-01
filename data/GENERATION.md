# Generation notes

- Seed: 42 (numpy default_rng) — fully reproducible
- N_USERS=5000, cold_start=500 (10%), established=4500
- Channel split: app/ussd = 50%/50%
- Benign stress sets (fixed volume): shared_device=150, sim_change=150
- Attacks (established): stolen_otp=50, drip=25, sim_swap=25
- Attacks (cold_start overlay): stolen_otp=5, drip=3, sim_swap=3
- Static fallback threshold used to cap drip escalation: 50,000 NGN
- Amounts: lognormal per-user baseline, session amounts lognormal around that baseline
- recipient_is_new / is_new_device: derived chronologically per user, not randomized independently
- Design simplification: 1 transaction per session (documented, not an oversight)
