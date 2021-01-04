import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from utils.model_utils import build_model, encode_categories
from utils.io_utils import yaml_loader, check_if_exists

config = yaml_loader("./config/config.yml")
np.random.seed(config["model"]["random_seed"])
tf.random.set_seed(config["model"]["random_seed"])

model_dataframe = pd.read_csv(config["paths"]["model_dataframe"])

encoded_dataframe = encode_categories(model_dataframe, config)

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
    dataframe=encoded_dataframe,
    x_col=config["model"]["paths"],
    y_col=config["model"]["target"],
    class_mode=config["model"]["class_mode"],
    shuffle=True,
    batch_size=config["model"]["batch_size"],
    subset="training",
    seed=config["model"]["random_seed"],
)
test_generator = image_generator.flow_from_dataframe(
    dataframe=encoded_dataframe,
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

early_stopping = EarlyStopping(
    patience=config["model"]["early_stopping_patience"],
    restore_best_weights=True,
    verbose=2,
)

LR_reducer = ReduceLROnPlateau(
    monitor="val_loss",
    factor=config["model"]["lr_reducer_factor"],
    patience=config["model"]["lr_reducer_patience"],
    min_lr=config["model"]["min_lr"],
    verbose=2,
)

csv_logger = CSVLogger(config["model"]["training_history_path"])

model.fit(
    x=train_generator,
    validation_data=test_generator,
    epochs=100,
    callbacks=[early_stopping, LR_reducer, csv_logger],
)

model_path = config["paths"]["model_path"]
check_if_exists(os.path.dirname(model_path), create=True)
model.save(model_path)
print(f"Model saved to {model_path}")
