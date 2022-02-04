import cv2


def include_text(image, bbox, tag, colour=[255,255,255], pos='bottom'):
    # Put text at the bottom of bounding box
    (text_width, text_height), baseline = cv2.getTextSize(
        tag, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    )
    bbox_width = int(bbox[2] - bbox[0])
    if pos == 'bottom':
        offset = int((bbox_width - text_width) / 2)
        position = (bbox[0] + offset, bbox[3] + text_height + baseline)
    elif pos == 'top_left':
        position = (bbox[0]-text_width, bbox[1] - baseline)
    elif pos == 'top_right':
        position = (bbox[2], bbox[1] - baseline)
    cv2.putText(
        image, tag, position, cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 3
    )
