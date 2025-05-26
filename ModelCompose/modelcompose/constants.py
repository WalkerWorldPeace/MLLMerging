CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Modal Constants
MODAL_TOKENS = {
    'vision': DEFAULT_IMAGE_TOKEN,
    'relrep': "<relrep>",
    'text': "<text>",
    'audio': "<audio>",
    'video': "<video>",
    'point': "<point>"
}
MODAL_TOKEN_INDEXES = {
    'vision': -200,
    'relrep': -201,
    'text': -202,
    'audio': -203,
    'video': -204,
    'point': -205
}
MODAL_TOKEN_MAPPING = {MODAL_TOKENS[k]: MODAL_TOKEN_INDEXES[k] for k in MODAL_TOKENS}

# Checkpoint paths
# Note: Please modify the constants below to your checkpoint paths.
VIDEO_CONFIG_PATH = "modelcompose/model/multimodal_encoder/LanguageBind_Video_merge"
MODEL_BASE = "vicuna-7b-v1.5"