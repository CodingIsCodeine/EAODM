"""
Default configuration for M²MVT.
Merged with the per-run YAML before training.
"""

from fvcore.common.config import CfgNode as CN

_C = CN()

# ── Training ──────────────────────────────────────────────────────────────────
_C.TRAIN = CN()
_C.TRAIN.ENABLE            = True
_C.TRAIN.DATASET           = "daad"
_C.TRAIN.BATCH_SIZE        = 16
_C.TRAIN.EVAL_PERIOD       = 10
_C.TRAIN.CHECKPOINT_PERIOD = 10
_C.TRAIN.AUTO_RESUME       = True

# ── Data ──────────────────────────────────────────────────────────────────────
_C.DATA = CN()
_C.DATA.PATH_TO_DATA_DIR          = ""
_C.DATA.NUM_FRAMES                = 16
_C.DATA.SAMPLING_RATE             = 4
_C.DATA.TRAIN_JITTER_SCALES       = [256, 320]
_C.DATA.TRAIN_CROP_SIZE           = 224
_C.DATA.TEST_CROP_SIZE            = 224
_C.DATA.INPUT_CHANNEL_NUM         = [3]
_C.DATA.DECODING_BACKEND          = "torchvision"
_C.DATA.USE_OFFSET_SAMPLING       = True
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = [0.08, 1.0]
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = [0.75, 1.3333]

# ── MViT ─────────────────────────────────────────────────────────────────────
_C.MVIT = CN()
_C.MVIT.ZERO_DECAY_POS_CLS     = False
_C.MVIT.USE_ABS_POS             = False
_C.MVIT.REL_POS_SPATIAL         = True
_C.MVIT.REL_POS_TEMPORAL        = True
_C.MVIT.DEPTH                   = 16
_C.MVIT.NUM_HEADS               = 1
_C.MVIT.EMBED_DIM               = 96
_C.MVIT.PATCH_KERNEL            = [3, 7, 7]
_C.MVIT.PATCH_STRIDE            = [2, 4, 4]
_C.MVIT.PATCH_PADDING           = [1, 3, 3]
_C.MVIT.MLP_RATIO               = 4.0
_C.MVIT.QKV_BIAS                = True
_C.MVIT.DROPPATH_RATE           = 0.2
_C.MVIT.NORM                    = "layernorm"
_C.MVIT.MODE                    = "conv"
_C.MVIT.CLS_EMBED_ON            = True
_C.MVIT.DIM_MUL                 = [[1, 2.0], [3, 2.0], [14, 2.0]]
_C.MVIT.HEAD_MUL                = [[1, 2.0], [3, 2.0], [14, 2.0]]
_C.MVIT.POOL_KVQ_KERNEL         = [3, 3, 3]
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 8, 8]
_C.MVIT.POOL_Q_STRIDE           = [
    [0,  1, 1, 1], [1,  1, 2, 2], [2,  1, 1, 1], [3,  1, 2, 2],
    [4,  1, 1, 1], [5,  1, 1, 1], [6,  1, 1, 1], [7,  1, 1, 1],
    [8,  1, 1, 1], [9,  1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1],
    [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1],
]
_C.MVIT.DROPOUT_RATE            = 0.0
_C.MVIT.DIM_MUL_IN_ATT         = True
_C.MVIT.RESIDUAL_POOLING        = True

# ── Augmentation ──────────────────────────────────────────────────────────────
_C.AUG = CN()
_C.AUG.NUM_SAMPLE   = 2
_C.AUG.ENABLE       = True
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AA_TYPE      = "rand-m7-n4-mstd0.5-inc1"
_C.AUG.INTERPOLATION = "bicubic"
_C.AUG.RE_PROB      = 0.25
_C.AUG.RE_MODE      = "pixel"
_C.AUG.RE_COUNT     = 1
_C.AUG.RE_SPLIT     = False

# ── Mixup ─────────────────────────────────────────────────────────────────────
_C.MIXUP = CN()
_C.MIXUP.ENABLE            = True
_C.MIXUP.ALPHA             = 0.8
_C.MIXUP.CUTMIX_ALPHA      = 1.0
_C.MIXUP.PROB              = 1.0
_C.MIXUP.SWITCH_PROB       = 0.5
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ── Solver ────────────────────────────────────────────────────────────────────
_C.SOLVER = CN()
_C.SOLVER.ZERO_WD_1D_PARAM          = True
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS  = True
_C.SOLVER.CLIP_GRAD_L2NORM          = 1.0
_C.SOLVER.BASE_LR                   = 1e-4
_C.SOLVER.COSINE_AFTER_WARMUP       = True
_C.SOLVER.COSINE_END_LR             = 1e-6
_C.SOLVER.WARMUP_START_LR           = 1e-6
_C.SOLVER.WARMUP_EPOCHS             = 30.0
_C.SOLVER.LR_POLICY                 = "cosine"
_C.SOLVER.MAX_EPOCH                 = 200
_C.SOLVER.MOMENTUM                  = 0.9
_C.SOLVER.WEIGHT_DECAY              = 0.05
_C.SOLVER.OPTIMIZING_METHOD         = "adamw"

# ── Model ─────────────────────────────────────────────────────────────────────
_C.MODEL = CN()
_C.MODEL.NUM_CLASSES   = 7
_C.MODEL.ARCH          = "mvit"
_C.MODEL.MODEL_NAME    = "M2MVT"
_C.MODEL.LOSS_FUNC     = "soft_cross_entropy"
_C.MODEL.DROPOUT_RATE  = 0.5

# ── Test ──────────────────────────────────────────────────────────────────────
_C.TEST = CN()
_C.TEST.ENABLE              = True
_C.TEST.DATASET             = "daad"
_C.TEST.BATCH_SIZE          = 64
_C.TEST.NUM_SPATIAL_CROPS   = 1
_C.TEST.NUM_ENSEMBLE_VIEWS  = 5

# ── DataLoader ────────────────────────────────────────────────────────────────
_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY  = True

# ── System ────────────────────────────────────────────────────────────────────
_C.NUM_GPUS   = 1
_C.NUM_SHARDS = 1
_C.RNG_SEED   = 0
_C.OUTPUT_DIR = "."

# ── Logging ───────────────────────────────────────────────────────────────────
_C.LOG_PERIOD = 10


def get_cfg():
    return _C.clone()
