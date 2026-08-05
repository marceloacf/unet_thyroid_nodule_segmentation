import os

# =========================
# Caminhos
# =========================

DATASET_PATH = "C:/Users/Marcelo/Documents/ProjetoTCC/Thyroid Dataset/tn3k"

EXPERIMENT_NAME = "experimento_3_kfold_32filtros"

EXPERIMENTS_DIR = os.path.join("outputs", "experiments")
EXPERIMENT_DIR = os.path.join(EXPERIMENTS_DIR, EXPERIMENT_NAME)

MODELS_DIR = os.path.join(EXPERIMENT_DIR, "models")
HISTORIES_DIR = os.path.join(EXPERIMENT_DIR, "histories")
OPTUNA_DIR = os.path.join(EXPERIMENT_DIR, "optuna")
KFOLD_DIR = os.path.join(EXPERIMENT_DIR, "kfold")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


# =========================
# Criação de diretórios
# =========================

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORIES_DIR, exist_ok=True)
os.makedirs(OPTUNA_DIR, exist_ok=True)
os.makedirs(KFOLD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# Reprodutibilidade
# =========================

SEED = 42


# =========================
# Dados
# =========================

TARGET_SIZE = 192
INPUT_SHAPE = (TARGET_SIZE, TARGET_SIZE, 1)

# Mantido por compatibilidade com experimentos antigos.
# No K-Fold, essa variável não será usada.
VALIDATION_SIZE = 0.15


# =========================
# Pré-processamento
# =========================

USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# =========================
# Data augmentation
# =========================

FLIP_LEFT_RIGHT = True
FLIP_UP_DOWN = False
BRIGHTNESS_DELTA = 0.08
CONTRAST_RANGE = (0.95, 1.05)
GAMMA_RANGE = (0.90, 1.10)
SPECKLE_STD = 0.03


# =========================
# K-Fold
# =========================

N_SPLITS = 5

FOLDS_CSV_PATH = os.path.join(
    KFOLD_DIR,
    "folds_stratified_area.csv"
)

AREA_SMALL_MAX = 0.01
AREA_MEDIUM_MAX = 0.05

# Para teste rápido, use [1].
# Para rodar todos os folds, use None.
#FOLDS_TO_RUN = [1]
FOLDS_TO_RUN = None
# FOLDS_TO_RUN = [1, 2, 3]


# =========================
# Treinamento K-Fold sem Optuna
# =========================

EPOCHS = 100

N_FILTERS = 32
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1

BLOCKS_PER_STAGE = (3, 4, 6, 3)
#3,4,6,3 resnet34

# =========================
# Callbacks
# =========================

EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4

REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN_DELTA = 1e-4
REDUCE_LR_COOLDOWN = 2


# =========================
# Optuna
# =========================
# Mantido para uso futuro. Não será usado inicialmente no K-Fold sem Optuna.
# =========================

N_TRIALS = 40

N_FILTERS_OPTIONS = [32, 64]
BATCH_SIZE_OPTIONS = [4, 8]

LEARNING_RATE_MIN = 1e-7
LEARNING_RATE_MAX = 1e-2

DROPOUT_RATE_MIN = 0.0
DROPOUT_RATE_MAX = 0.3

STUDY_NAME = EXPERIMENT_NAME
OPTUNA_STORAGE = f"sqlite:///{os.path.join(OPTUNA_DIR, 'study.db')}"


# =========================
# Avaliação
# =========================

THRESHOLD = 0.5