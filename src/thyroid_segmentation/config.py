import os

# =========================
# Caminhos
# =========================

DATASET_PATH = "C:/Users/marcelo/Documents/projetoTcc/Thyroid Dataset/tn3k"

EXPERIMENT_NAME = "melhormodeloseg"
#sem o attention gate com numfilters//2
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

TARGET_SIZE = 160
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

AREA_SMALL_MAX = 0.05
AREA_MEDIUM_MAX = 0.20

# Para teste rápido, use [1].
# Para rodar todos os folds, use None.
#FOLDS_TO_RUN = [1]
FOLDS_TO_RUN = None
# FOLDS_TO_RUN = [1, 2, 3]


# =========================
# Treinamento K-Fold sem Optuna
# =========================

EPOCHS = 100

N_FILTERS = 64
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.1
USE_ATTENTION = True

BLOCKS_PER_STAGE = (3, 4, 6, 3)
#3,4,6,3 resnet34

# =========================
# Callbacks
# =========================

EARLY_STOPPING_PATIENCE = 40
EARLY_STOPPING_MIN_DELTA = 1e-4

REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN_DELTA = 1e-4
REDUCE_LR_COOLDOWN = 2


# =========================
# Visualização
# =========================

# Fold cujo modelo será carregado.
VISUALIZATION_FOLD = 1

# Quantidade de imagens a mostrar quando VISUALIZATION_INDICES = None.
VISUALIZATION_NUM_SAMPLES = 15

# Opções:
# "random" -> amostra aleatória reprodutível
# "first"  -> primeiras imagens do conjunto de teste
# "best"   -> maiores Dice
# "worst"  -> menores Dice
VISUALIZATION_MODE = "random"

# Para visualizar casos específicos, informe uma lista.
# Exemplo:
# VISUALIZATION_INDICES = [600, 516, 443, 87]
#
# Quando diferente de None, esta lista tem prioridade sobre
# VISUALIZATION_MODE e VISUALIZATION_NUM_SAMPLES.
VISUALIZATION_INDICES = None

# Quantas imagens aparecem em cada janela/arquivo gerado.
# Evita criar uma figura enorme quando você pede 30, 50 ou mais casos.
VISUALIZATION_SAMPLES_PER_FIGURE = 5

# Salva as figuras em:
# outputs/experiments/<EXPERIMENT_NAME>/results/visualization/fold_<N>/
VISUALIZATION_SAVE = True

# Abre as janelas do Matplotlib durante a execução local.
VISUALIZATION_SHOW = True

# Mostra uma quarta coluna com os contornos:
# ground truth = verde
# predição = vermelho
VISUALIZATION_SHOW_OVERLAY = True


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