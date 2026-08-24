FEATURE_COLUMNS = [
    "loss",
    "grad_norm_l2",
    "grad_cosine_past",
]

STAGE_GRAD_FEATURE_COLUMNS = [
    "grad_norm_stage_stem",
    "grad_norm_stage_layer1",
    "grad_norm_stage_layer2",
    "grad_norm_stage_layer3",
    "grad_norm_stage_layer4",
]

FEATURE_COLUMNS = (
        FEATURE_COLUMNS
        + STAGE_GRAD_FEATURE_COLUMNS
)

FEATURE_PROTOCOL = "clean_supervised_stage_gradient_v1"

HEAD_MODE = "defender_fixed"