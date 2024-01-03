# utils.py

import torch
from torchvision import transforms, datasets

def load_data(data_dir):
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets
    image_datasets = {x: datasets.ImageFolder(root=data_dir + x,                    transform=data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    # Define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'valid', 'test']}

    return dataloaders, image_datasets
def process_image(image_path):
    # Charger l'image avec PIL
    image = Image.open(image_path)

    # Appliquer les transformations nécessaires
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = preprocess(image)

    # Ajouter la dimension du batch
    img_tensor = img_tensor.unsqueeze(0)

    # Retourner le tableau NumPy
    return img_tensor.numpy()