_base_ = "./mini_raydn_b0_common_50.py"

work_dir = "outputs/stage3/raydn_screening/b0_raydn_50"
model = dict(
    pts_bbox_head=dict(
        type="RayDNStreamPETRHead",
        enable_ray_denoising=True,
        raydn_group=1,
        raydn_num=5,
        raydn_alpha=8.0,
        raydn_beta=2.0,
        raydn_radius=3.0,
    ),
)

