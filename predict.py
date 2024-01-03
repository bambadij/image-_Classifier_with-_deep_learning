# predict.py

import argparse
from model import load_checkpoint
from utils import process_image
import json
import torch
from torch.autograd import Variable

def predict(image_path, checkpoint_path, arch='vgg16', topk=5, gpu=False,category_names=None):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Load the checkpoint
    model = load_checkpoint(checkpoint_path, arch)

    # Process the image
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze(0).float()

    # Move model and image tensor to the appropriate device
    model.to(device)
    image = image.to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Calculate probabilities and class indices
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probs, topk)
    
    #load category names from JSON file
    class_names =None
    if category_names:
        with open(category_names,'r') as f:
            class_names = json.load(f)
    #convert class indices to category names
    if class_names:
        top_classes = [class_names[str(idx)] for idx in top_indices.cpu().numpy()]
    else:
        top_classes = top_indices.cpu().numpy()
    return top_probs.cpu().numpy(), top_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class for an input image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file")
    parser.add_argument("--arch", default="vgg16", help="Model architecture")
    parser.add_argument("--topk", type=int, default=5, help="Number of top classes to show")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--category_names", help="Path to the JSON file containing category names")

    args = parser.parse_args()

    # Perform prediction
    probs, classes = predict(args.image_path, args.checkpoint_path, args.arch, args.topk, args.gpu, args.category_names)

    # Print the results
    print("Probabilities:", probs)