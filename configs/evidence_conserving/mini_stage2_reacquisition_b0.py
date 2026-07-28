"""S2.3 rescue B0: unchanged S2.2 source-ledger reference."""

_base_ = "./mini_stage2_source_ledger_debug_50.py"

work_dir = "outputs/stage2/s2_3_rescue/zero_shot/b0"
load_from = (
    "/home/research/research/evidence3d/outputs/stage2/"
    "s2_2_source_ledger_debug_50/iter_50.pth"
)
resume_from = None
