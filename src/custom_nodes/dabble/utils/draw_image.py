import cv2
import numpy as np
from typing import Union, List


def include_text(
    image: np.ndarray,
    bbox: Union[np.ndarray, list],
    tag: str,
    colour=[255, 255, 255],
    pos="bottom",
):
    """To add text to the various locations of the bbox

    Args:
        image (np.ndarray): Image to be written on
        bbox (Union[np.ndarray, list]): Bounding box to put the text on
        tag (str): The text to write on the image
        colour (list, optional): Colour of text. Defaults to [255,255,255].
        pos (str, optional): Position of text. Defaults to 'bottom'.
    """
    (text_width, text_height), baseline = cv2.getTextSize(
        tag, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    )
    bbox_width = int(bbox[2] - bbox[0])
    if pos == "bottom":
        offset = int((bbox_width - text_width) / 2)
        position = (bbox[0] + offset, bbox[3] + text_height + baseline)
    elif pos == "top_left":
        position = (bbox[0] - text_width, bbox[1] - baseline)
    elif pos == "top_left_upper":
        position = (bbox[0] - text_width, bbox[1] - baseline * 2 - text_height)
    elif pos == "top_right":
        position = (bbox[2], bbox[1] - baseline)
    cv2.putText(image, tag, position, cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3)


def bboxes_rescaling(
    bboxes: List[Union[list, tuple]], image: np.ndarray,
) -> List[Union[list, tuple]]:
    """Rescale the normalized bboxes to the original scale.

    Args:
        bboxes (List[Union[list, tuple]]): Bounding boxes to resize
        image (np.ndarray): Image of the bbox to be resized

    Returns:
        List[Union[list, tuple]]: Rescaled bboxes
    """
    bboxes_rescaled = []
    img_n_rows, img_n_cols, _ = image.shape
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * img_n_cols)
        x_max = int(x_max * img_n_cols)
        y_min = int(y_min * img_n_rows)
        y_max = int(y_max * img_n_rows)
        bboxes_rescaled.append([x_min, y_min, x_max, y_max])

    return bboxes_rescaled
