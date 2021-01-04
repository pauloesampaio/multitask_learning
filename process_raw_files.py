import pandas as pd
from utils.dataset_utils import (
    split_list,
    mount_dataset,
    process_images,
)
from utils.io_utils import txt_loader, yaml_loader

config = yaml_loader("./config/dataset_config.yml")
category_list = split_list(txt_loader(config["paths"]["categories_path"], skip_lines=2))
category_dict = {w[0]: int(w[1]) for w in category_list}
attribute_list = split_list(
    txt_loader(config["paths"]["attributes_path"], skip_lines=2)
)
attribute_dict = {w[0]: int(w[1]) for w in attribute_list}
datasets = ["train", "test", "val"]
dataset_dict = {}
for dataset in datasets:
    dataset_dict[dataset] = mount_dataset(
        files_path=config["paths"][f"{dataset}_files_path"],
        categories_path=config["paths"][f"{dataset}_categories_path"],
        attributes_path=config["paths"][f"{dataset}_attributes_path"],
        bboxes_path=config["paths"][f"{dataset}_bboxes_path"],
        category_dict=category_dict,
        attribute_dict=attribute_dict,
    )
full_dataset = pd.concat(dataset_dict.values()).reset_index(drop=True)
full_dataset["file"] = [
    f'{config["paths"]["data_folder_prefix"]}/{w}' for w in full_dataset["file"]
]

full_dataset.to_parquet(config["paths"]["full_dataset_path"], index=False)

category_translator = {}
for k in config["categories"]["aggregation"].keys():
    for v in config["categories"]["aggregation"][k]:
        category_translator[v] = k
full_dataset["top_level_category"] = full_dataset["category"].map(category_translator)
model_subset = full_dataset.loc[
    full_dataset["top_level_category"].isin(config["categories"]["to_model"]), :
].reset_index(drop=True)
model_subset.to_parquet(config["paths"]["model_subset_path"], index=False)

processed_subset = process_images(model_subset, config["paths"]["cropped_folder"])
processed_subset = processed_subset.dropna(subset=["processed_path"])
processed_subset = processed_subset.drop(columns=["bbox"])
processed_subset.to_csv(config["paths"]["processed_subset_path"], index=False)
