import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from .io_utils import check_if_exists, txt_loader


def split_list(list_of_strings):
    return [[w for w in c.split(" ") if w != ""] for c in list_of_strings]


def str_to_int(list_of_strings):
    return [[int(w) for w in c] for c in list_of_strings]


def parse_categoricals(list_of_ints, categorical_dict, positional=False):
    categorical_list = list(categorical_dict.keys())
    if positional:
        parsed = [
            [categorical_list[i] for i, x in enumerate(c) if x == 1]
            for c in list_of_ints
        ]
    else:
        parsed = [categorical_list[c[0] - 1] for c in list_of_ints]
    return parsed


def mount_dataset(
    files_path,
    categories_path,
    attributes_path,
    bboxes_path,
    category_dict,
    attribute_dict,
):
    test_files = txt_loader(files_path)
    test_categories = str_to_int(split_list(txt_loader(categories_path)))
    test_attributes = str_to_int(split_list(txt_loader(attributes_path)))
    test_bboxes = str_to_int(split_list(txt_loader(bboxes_path)))
    parsed_categories = parse_categoricals(
        test_categories, category_dict, positional=False
    )
    parsed_attributes = parse_categoricals(
        test_attributes, attribute_dict, positional=True
    )
    dataset = pd.concat(
        (
            pd.DataFrame(
                zip(test_files, parsed_categories, test_bboxes),
                columns=["file", "category", "bbox"],
            ),
            pd.DataFrame(
                parsed_attributes,
                columns=["print", "sleeves", "length", "neckline", "fabric", "fit"],
            ),
        ),
        axis=1,
    )
    return dataset


def process_images(dataframe, processed_folder):
    output_dataframe = dataframe.copy()
    output_dataframe["processed_path"] = pd.NA
    check_if_exists(processed_folder, create=True)
    for i in tqdm(range(output_dataframe.shape[0])):
        input_path = output_dataframe.loc[i, "file"]
        output_path = [processed_folder] + input_path.split("/")[-2:]
        output_path = os.path.join(*output_path)
        check_if_exists(os.path.dirname(output_path), create=True)
        try:
            Image.open(input_path).crop(output_dataframe.iloc[i]["bbox"]).save(
                output_path
            )
            output_dataframe.loc[i, "processed_path"] = output_path
        except IOError:
            print(input_path)
    return output_dataframe
