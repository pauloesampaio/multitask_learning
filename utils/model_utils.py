from PIL import Image
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception, preprocess_input


def download_image(image_url):
    """Downloads image from an url and returns PIL image

    Args:
        image_url (str): url of the desires image

    Returns:
        PIL Image: downloaded image
    """
    resp = requests.get(image_url, stream=True, timeout=5)
    im_bytes = BytesIO(resp.content)
    image = Image.open(im_bytes)
    return image


def build_model(config):
    input_shape = config["model"]["input_shape"] + [3]
    i = Input(
        input_shape,
        name="model_input",
    )
    x = preprocess_input(i)
    core = Xception(input_shape=input_shape, include_top=False, pooling="avg")

    if config["model"]["freeze_convolutional_layers"]:
        print("Freezing convolutional layers")
        core.trainable = False

    x = core(x)
    outputs = []
    for clf_layer in config["model"]["target_encoder"]:
        n_classes = len(config["model"]["target_encoder"][clf_layer])
        outputs.append(
            Dense(units=n_classes, activation="softmax", name=f"{clf_layer}_clf")(x)
        )
    model = Model(inputs=i, outputs=outputs)
    return model


def predict(model, image, config):
    model_input = np.array(image.resize((224, 224))).reshape(1, 224, 224, 3)
    prediction = model.predict(model_input)
    prediction_dictionary = {}
    for i, encoder in enumerate(config["model"]["target_encoder"].keys()):
        label = config["model"]["target_encoder"][encoder][np.argmax(prediction[i])]
        probability = np.max(prediction[i])
        prediction_dictionary[encoder] = {}
        prediction_dictionary[encoder]["label"] = label
        prediction_dictionary[encoder]["probability"] = probability
    return prediction_dictionary
