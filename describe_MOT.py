#!/usr/bin/env python3
import glob
import os
import pickle
import sys

import skimage.io

import mrcnn.model as modellib
from mrcnn import utils
from samples.coco import coco

# Root directory of the project
ROOT_DIR = sys.path[0]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

folders = glob.glob(os.path.join(sys.argv[1], '*/*'))
for folder in folders:
    try:
        os.mkdir(os.path.join(folder, 'mrcnn'))
    except FileExistsError:
        pass
    files = glob.glob(os.path.join(folder, 'img1/*'))
    for file in files:
        print(file)
        image = skimage.io.imread(file)
        result = model.detect([image])[0]
        with open(os.path.join(folder, 'mrcnn', '{}.pickle'.format(os.path.basename(file))), 'wb') as handle:
            pickle.dump(result, handle)
