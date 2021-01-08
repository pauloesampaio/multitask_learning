import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from .io_utils import check_if_exists, txt_loader

# All these functions are specific for dealing with the deep fashion dataset


def split_list(list_of_strings):
    """Get a list of strings and splits this strings on spaces,
    keeping only non empty values.

    Args:
        list_of_strings (list): list of strings

    Returns:
        list: List of lists with no empty elements
    """
    return [[w for w in c.split(" ") if w != ""] for c in list_of_strings]


def str_to_int(list_of_strings):
    """Goes through lists of lists of number as strings and pass them to ints

    Args:
        list_of_strings (list): list of lists of numbers encoded as strings

    Returns:
        list: Processed list, with numbers as int
    """
    return [[int(w) for w in c] for c in list_of_strings]


def parse_categoricals(list_of_ints, categorical_dict, positional=False):
    """Receiving a list of encoded categories and having the category dictionary,
    convert the encoded to the actual values. If positional, infer the encoded
    category by the position of 1 in a binary list.

    Args:
        list_of_ints (list): Encoded category
        categorical_dict (dict): Dictionary that maps int to actual label
        positional (bool, optional): If true, infers encoded category by
        position of 1 in a binary list. Defaults to False.

    Returns:
        list: List of actual labels
    """
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
    """Very specific to the deep fashion dataset, this function
    receives the paths to the images, to the dataset labels and
    bounding boxes and the categories and attributes dictionaries.
    Returns a dataframe with file path and actual "human readable"
    labels.

    Args:
        files_path (str): Path to the images
        categories_path (str): Path to the txt file with categories for the images
        attributes_path (str): Path to the txt file with attributes for the images
        bboxes_path (str): Path to the txt file with bounding boxes coordinates for the images
        category_dict (dict): Dictionary that translate encoded categories to actual labels
        attribute_dict (dict): Dictionary that translate encoded attributes to actual labels

    Returns:
        pd.DataFrame: Dataframe with image paths and human-friendly labels
    """
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
    """Having a dataframe with image paths and bounding boxes coordinates,
    crops the bounding boxes and save the images.

    Args:
        dataframe (pd.DataFrame): Dataframe with file paths and bounding boxes coordinates
        processed_folder (str): Path where the cropped images should be stored

    Returns:
        pd.DataFrame: Same as input dataframe but with "processed_path" column
    """
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
