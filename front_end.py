import streamlit as st
import pandas as pd
from utils.io_utils import download_image, yaml_loader
from utils.model_utils import predict
from tensorflow.keras.models import load_model

config = yaml_loader("./config/config.yml")


@st.cache
def model_loader(config):
    model = load_model(config["paths"]["model_path"])
    return model


model = model_loader(config)

st.write(
    """
# Multi task classifier
## Enter the image url
"""
)
url = st.text_input("Enter image url")
if url:
    current_image = download_image(url)
    predictions = predict(model, current_image, config)
    col1, col2 = st.beta_columns(2)
    col1.write("Original image")
    col1.image(current_image, use_column_width=True)
    for encoder in predictions.keys():
        current_dict = predictions[encoder]
        current_df = pd.DataFrame(
            index=current_dict.keys(),
            data=current_dict.values(),
            columns=["prediction"],
        ).sort_values(by="prediction", ascending=False)
        col2.write(f"{encoder} prediction")
        col2.dataframe(current_df)
