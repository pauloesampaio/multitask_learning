import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception, preprocess_input


def build_model(config):
    """Builds a keras model using Xception network as core and following
    instructions on the configuration file.

    Args:
        config (dict): Configuration dictionary

    Returns:
        keras.model: Keras model
    """
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


def encode_categories(dataframe, config):
    """Function to one-hot-encode labels according to encoding map
    defined on the config file

    Args:
        dataframe (pd.DataFrame): Dataframe with actual labels
        config (dict): Configuration dictionary with encoding map

    Returns:
        pd.DataFrame: Dataframe with one-hot-encoded labels
    """
    for k in config["model"]["target_encoder"].keys():
        n_classes = len(config["model"]["target_encoder"][k])
        encoder = dict(zip(config["model"]["target_encoder"][k], range(n_classes)))
        dataframe[f"{k}_encoded"] = to_categorical(
            dataframe[f"{k}"].map(encoder), num_classes=n_classes
        ).tolist()
    return dataframe


def predict(model, image, config):
    """Prediction function, to not only predict but also to decode
    prediction into human friendly labels

    Args:
        model (keras.Model): Trained keras model
        image (PIL.Image): Image in PIL format
        config (dict): Configuration dictionary with key to decode predictions

    Returns:
        dict: Dictionary with predictions and probabilities
    """
    input_shape = config["model"]["input_shape"]
    model_input = np.array(image.resize(input_shape))
    model_input = model_input.reshape([1] + input_shape + [3])
    prediction = model.predict(model_input)
    prediction_dictionary = {}

    for i, encoder in enumerate(config["model"]["target_encoder"].keys()):
        labels = config["model"]["target_encoder"][encoder]
        probabilities = prediction[i][0]
        prediction_dictionary[encoder] = dict(zip(labels, probabilities))
    return prediction_dictionary
