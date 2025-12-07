# Model loading

import torch
import torchvision.models as models
from joblib import load


def load_model(model_path, params_path, device):
    # Load preprocessing params
    params = load(params_path)

    # Initialize EfficientNet-B0
    model = models.efficientnet_b0(weights=None)
    num_classes = len(params['classes'])
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Remove 'model.' prefix from keys if present
    if isinstance(checkpoint, dict):
        state_dict = {}
        for key, value in checkpoint.items():
            # Remove 'model.' prefix
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            state_dict[new_key] = value

        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    return model, params