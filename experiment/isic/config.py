"""
Variables set by the command line arguments dictating which parts of the program to execute.
Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
"""

# Constants
RANDOM_SEED = 111
MINI_MIAS_IMG_SIZE = {
    "HEIGHT": 1024,
    "WIDTH": 1024
}
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512
}
RESNET_IMG_SIZE = VGG_IMG_SIZE
INCEPTION_IMG_SIZE = VGG_IMG_SIZE
DENSE_NET_IMG_SIZE = VGG_IMG_SIZE
MOBILE_NET_IMG_SIZE = VGG_IMG_SIZE
XCEPTION_IMG_SIZE = INCEPTION_IMG_SIZE
ROI_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}

# Variables set by command line arguments/flags
dataset = "mini-MIAS"       # The dataset to use.
mammogram_type = "all"      # The type of mammogram (Calc or Mass).
model = "ResNet"               # The model to use.
run_mode = "training"       # The type of running mode, either training or testing.
learning_rate = 1e-3        # The learning rate with the pre-trained ImageNet layers frozen.
batch_size = 2              # Batch size.
max_epoch_frozen = 10      # Max number of epochs when original CNN layers are frozen.
max_epoch_unfrozen = 10     # Max number of epochs when original CNN layers are unfrozen.
is_roi = True              # Use cropped version of the images
verbose_mode = True      # Boolean used to print additional logs for debugging purposes.
name = ""                   # Name of experiment.
# is_grid_search = False    # Run the grid search algorithm to determine the optimal hyper-parameters for the model.

#CAT-KD config
IF_NORMALIZE = True
onlyCAT = False
IF_BINARIZE = False

IF_OnlyTransferPartialCAMs = False
CAMs_Nums = 100
Strategy = 0

# LOSS = CN()
CE_WEIGHT = 1.0
CAT_loss_weight = 400.0
CAM_RESOLUTION = 2

teacher_dir = None
student_dir = None