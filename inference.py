import os
import cv2
import torch
import numpy as np
import yaml
from models.create_fasterrcnn_model import create_model
from utils.transforms import infer_transforms, resize
from utils.annotations import convert_detections, inference_annotations
import glob


def run_inference(image_path, args):
    """
    Run inference on a single image.

    :param image_path: Path to the input image.
    :param args: Arguments for inference (e.g., model, weights, etc.).
    :return: Processed image with bounding boxes.
    """
    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']

    DEVICE = args['device']

    # Load the model
    checkpoint = torch.load(args['weights'], map_location=DEVICE)
    build_model = create_model['fasterrcnn_resnet50_fpn_v2']
    model = build_model(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()


    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    DIR_TEST = image_path

    test_images = collect_all_images(DIR_TEST)
    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        if args['imgsz'] != None:
            RESIZE_TO = args['imgsz']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(
            orig_image, RESIZE_TO, square=args['square_img']
        )
        image = image_resized.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image.to(DEVICE))

        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        pred_classes_list = []  # List to store predicted classes

        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            draw_boxes, pred_classes, scores, labels = convert_detections(
                outputs, args['threshold'], CLASSES, args
            )

            filtered_classes = [
                pred_classes[j] for j in range(len(scores)) if scores[j] >= args['threshold']
            ]

            pred_classes_list.extend(filtered_classes)  # Collect predicted classes
            orig_image = inference_annotations(
                draw_boxes, 
                pred_classes, 
                scores,
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )

    return orig_image, pred_classes_list

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    