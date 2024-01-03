# train.py

import argparse
from model import load_checkpoint
from utils import load_data
from torch import optim, nn
from torch.autograd import Variable

def train_model(data_dir, save_dir, arch='vgg16', hidden_units=512, learning_rate=0.01, epochs=1, gpu=False):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Load data
    dataloaders, _ = load_data(data_dir)

    # Build model
    model = load_checkpoint(arch, hidden_units)

    # Define loss criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move model to the appropriate device
    model.to(device)

    # Train the model
   for epoch in range(epochs):
        model.train()

        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate training loss
        train_loss = running_loss / dataset_sizes['train']

        # Validate the model
        model.eval()
        val_loss = 0
        correct = 0
        total_size=0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total_size+=labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate validation loss and accuracy
        val_loss /= dataset_sizes['valid']
        val_accuracy = correct / dataset_sizes['valid']

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f} "
              f"Validation loss: {val_loss:.3f} "
              f"Validation accuracy: {val_accuracy:.3f}")

    # Save the checkpoint
    save_checkpoint(model, save_dir, arch, hidden_units)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument("data_dir", help="Path to the dataset")
    parser.add_argument("--save_dir", default="checkpoint.pth", help="Directory to save checkpoints")
    parser.add_argument("--arch", default="vgg16", help="Model architecture")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()
  
