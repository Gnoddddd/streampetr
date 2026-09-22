"""Frozen Stage 4 V1 protocol for the first formal GeoCorr experiment."""

formal_training = dict(
    manifest="outputs/geocorr_stage3a/manifests/train_dirt_water.jsonl",
    top_k=25,
    epochs=5,
    learning_rate=1e-4,
    lambda_corr=1.0,
    lambda_rec=1.0,
    grad_accum=1,
    shuffle=True,
    seed=20260922,
    trainable=("correlation.adapter", "recovery.recovery"),
    frozen_detector=True,
)

formal_evaluation = dict(
    manifest="outputs/geocorr_stage3a/manifests/val_dirt_water.jsonl",
    protocol="OccNuScenes 4-scene official-val subset",
    top_k=25,
    modes=("baseline", "geocorr"),
    evaluate_clean=True,
    report_metrics=("mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"),
)
