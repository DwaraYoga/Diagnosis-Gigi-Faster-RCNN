import torchinfo
import torch
from models.create_fasterrcnn_model import create_model

def load_model(weights_path, model_name='fasterrcnn_resnet50_fpn_v2', num_classes=91, device='cpu'):
    """
    Load a model from a checkpoint file.

    :param weights_path: Path to the checkpoint file.
    :param model_name: Name of the model architecture.
    :param num_classes: Number of classes for the model.
    :param device: Device to load the model on ('cpu' or 'cuda').
    :return: Loaded model.
    """
    # Create the model
    build_model = create_model[model_name]
    model = build_model(num_classes=num_classes, coco_model=False)

    # Load the checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model

def summary(model):
    """
    Display the summary of the model.

    :param model: The model to summarize.
    """
    # Torchvision Faster RCNN models are enclosed within a tuple ().
    if type(model) == tuple:
        model = model[0]
    device = 'cpu'
    batch_size = 4
    channels = 3
    img_height = 640
    img_width = 640
    torchinfo.summary(
        model, 
        device=device, 
        input_size=[batch_size, channels, img_height, img_width],
        row_settings=["var_names"]
    )

if __name__ == "__main__":
    # Example usage
    weights_path = "best_model.pth"  # Replace with your checkpoint path
    model = load_model(weights_path)
    summary(model)