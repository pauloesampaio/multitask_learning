import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from utils.model_utils import build_model
from utils.io_utils import yaml_loader, check_if_exists

config = yaml_loader("./config/config.yml")
np.random.seed(config["model"]["random_seed"])
tf.random.set_seed(config["model"]["random_seed"])

processed_subset = pd.read_parquet(config["paths"]["processed_subset_path"])
image_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=[0.8, 1.2],
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    validation_split=0.2,
)
train_generator = image_generator.flow_from_dataframe(
    dataframe=processed_subset,
    x_col=config["model"]["paths"],
    y_col=config["model"]["target"],
    class_mode=config["model"]["class_mode"],
    shuffle=True,
    batch_size=config["model"]["batch_size"],
    subset="training",
    seed=config["model"]["random_seed"],
)
test_generator = image_generator.flow_from_dataframe(
    dataframe=processed_subset,
    x_col=config["model"]["paths"],
    y_col=config["model"]["target"],
    class_mode=config["model"]["class_mode"],
    shuffle=False,
    batch_size=config["model"]["batch_size"],
    subset="validation",
    seed=config["model"]["random_seed"],
)

model = build_model(config)
adam = Adam(config["model"]["learning_rate"])
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
logdir = os.path.join("./logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = TensorBoard(logdir, histogram_freq=1)


model.fit(
    x=train_generator,
    validation_data=test_generator,
    epochs=100,
    callbacks=[early_stopping, tensorboard],
)

model_path = config["paths"]["model_path"]
check_if_exists(os.path.dirname(model_path), create=True)
model.save(model_path)
print(f"Model saved to {model_path}")
