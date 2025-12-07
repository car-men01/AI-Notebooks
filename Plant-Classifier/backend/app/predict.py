# Inference logic for the machine learning model

import torch
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image: Image.Image, params):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(params['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=params['mean'], std=params['std'])
    ])
    return transform(image).unsqueeze(0)


def predict_image(image_bytes, package):
    model = package['model']
    params = package['params']
    device = package['device']

    # Load and preprocess image
    image = Image.open(image_bytes).convert('RGB')
    input_tensor = preprocess_image(image, params).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)

    results = {
        'predicted_class': params['classes'][top3_idx[0][0].item()],
        'confidence': top3_prob[0][0].item(),
        'top_3_predictions': [
            {'class': params['classes'][idx.item()], 'confidence': prob.item()}
            for prob, idx in zip(top3_prob[0], top3_idx[0])
        ]
    }
    return results