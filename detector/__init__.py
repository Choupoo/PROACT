BASE_FEATURE_COLUMNS = [
    "loss",
    "grad_norm_l2",
    "grad_cosine_past",
]

STAGE_NAMES = ("stem", "layer1", "layer2", "layer3", "layer4")
STAGE_GRAD_FEATURE_COLUMNS = [
    "grad_norm_stage_{}".format(stage) for stage in STAGE_NAMES
]
TASK_COSINE_FEATURE_COLUMNS = [
    "grad_cosine_task_min",
    "grad_cosine_task_max",
    "grad_cosine_task_mean",
]
UNCERTAINTY_FEATURE_COLUMNS = [
    "entropy",
    "confidence",
    "true_class_probability",
    "margin",
]
ACTIVATION_FEATURE_COLUMNS = ["activation_norm_l2"]
FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + TASK_COSINE_FEATURE_COLUMNS
    + UNCERTAINTY_FEATURE_COLUMNS
    + ACTIVATION_FEATURE_COLUMNS
    + STAGE_GRAD_FEATURE_COLUMNS
)

GRAD_PARAM_PREFIX = "grad_norm_param__"
GRAD_LAYER_PREFIX = "grad_norm_layer__"

FEATURE_PROTOCOL = "pretraining_full_v2"

HEAD_MODE = "defender_fixed"