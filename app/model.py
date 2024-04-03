import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from time import sleep
import os
from urllib.parse import urlparse
import sys

def search_and_download_images(searches, path, max_images=5):
    from duckduckgo_search import DDGS

    def search_images(term, max_images):
        print(f"Searching for '{term}'")
        urls = []

        with DDGS() as ddgs:
            ddgs_images_gen = ddgs.images(
                term,
                region="wt-wt",
                safesearch="off",
                size=None,
                color=None,
                type_image=None,
                layout=None,
                license_image=None,
                max_results=max_images,
            )

            for result in ddgs_images_gen:
                if 'image' in result:
                    urls.append(result['image'])

        return urls

    def download_and_resize_images(dest, urls, max_size=400):
        dest.mkdir(exist_ok=True, parents=True)
        for url in urls:
            try:
                response = requests.get(url, stream=True)
                content_type = response.headers.get('Content-Type')
                print(f"Downloading {url} ({content_type})")
                if 'image' in content_type:
                    img_data = response.content
                    response.close()
                    img = Image.open(BytesIO(img_data))
                    img = img.convert('RGB')
                    
                    parsed_url = urlparse(url)
                    clean_filename = os.path.basename(parsed_url.path)
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    img_path = dest / clean_filename
                    img.save(img_path)
                else:
                    print(f"URL did not point to an image: {url}")
            except Exception as e:
                print(f"Failed to download or resize image {url}: {e}")
    
    def verify_images(dir_path):
        failed = []
        for img_path in dir_path.iterdir():
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupt image: {img_path}")
                failed.append(img_path)
                img_path.unlink()
        return failed
    
    for term in searches:
        dest = path / term
        urls = search_images(f'{term} photo -seeds', max_images)
        download_and_resize_images(dest, urls)
        sleep(10)
    
    failed = verify_images(dest)
    print(f"Failed: {len(failed)}")

def get_dataloaders(path, batch_size=32):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=path, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_model(path, num_epochs=10):
    train_loader, val_loader = get_dataloaders(path)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), 'pc_model.pth')

def predict_image(model_path, image_path):
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Load model state dict correctly
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Call eval on the model instance
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    classes = ['coriander', 'parsley']
    print(f'Image: {image_path} - Prediction: {classes[preds[0].item()]}; Probability: {probs[preds[0]].item()}')

    # Return JSON of prediction
    return {'prediction': classes[preds[0].item()], 'probability': probs[preds[0]].item(), 'image': image_path}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on specified images or predict using an existing model.")
    parser.add_argument('--train', action='store_true', help='Download images and train the model.')
    parser.add_argument('--predict', type=str, help='Predict an image. Requires path to image.', metavar='IMAGE_PATH')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train the model.', metavar='EPOCHS', default=10)
    parser.add_argument('--train-path', type=str, help='Path to store training images.', metavar='TRAIN_PATH', default='~/')

    args = parser.parse_args()

    if args.train:
        search_terms = ['coriander', 'parsley']
        path = Path(args.train_path).expanduser()
        search_and_download_images(search_terms, path)
        train_model(path, args.epochs)

    elif args.predict:
        image_path = args.predict
        model_path = 'pc_model.pth'
        if not Path(model_path).exists():
            print(f"Model file {model_path} not found. Please train the model first.", file=sys.stderr)
            sys.exit(1)
        predict_image(model_path, image_path)
    else:
        parser.print_help()