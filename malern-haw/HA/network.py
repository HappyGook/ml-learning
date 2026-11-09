import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim


class FashionCNN(nn.Module):
    def __init__(self, layer_1_out: int = 32):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, layer_1_out, 3)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(layer_1_out, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def training(n_epochs: int, train_set, checkpoint_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_epoch = 1
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, n_epochs):
        print(f"Epoch {epoch}, training...")
        model.train()
        running_loss = 0.0

        for i, (inputs,labels) in enumerate(train_set):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished training")

    save_path = checkpoint_path or "fashion_cnn_checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": n_epochs,
    }, save_path)

    print(f"Finished training. Checkpoint saved to {save_path}")
    return model

def evaluate(test_set, model, fashion_classes, device=None):
    device = device or next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    # Rows are true classes; columns are predicted classes.
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():
        for inputs, labels in test_set:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * labels.size(0)

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for truth, guess in zip(labels.cpu(), predicted.cpu()):
                confusion[truth, guess] += 1

    accuracy = correct / total
    avg_loss = total_loss / total

    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test loss: {avg_loss:.4f}")
    print("\nPer-class metrics:")

    for class_id, name in enumerate(fashion_classes):
        true_positive = confusion[class_id, class_id].item()
        predicted_positive = confusion[:, class_id].sum().item()
        actual_positive = confusion[class_id, :].sum().item()

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        print(
            f"{name:12} precision: {precision:.2%}  "
            f"recall: {recall:.2%}  F1: {f1:.2%}"
        )

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "confusion_matrix": confusion,
    }


def plot_examples(test_set, fashion_classes, model, device=None, count=3, ):
    device = device or next(model.parameters()).device
    dataset = test_set.dataset

    if len(dataset) < count:
        raise ValueError(f"The test dataset has fewer than {count} examples.")

    model.eval()
    indices = torch.randperm(len(dataset))[:count].tolist()

    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4))
    if count == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, index in zip(axes, indices):
            image, label = dataset[index]
            output = model(image.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).item()

            # ToTensor() images have shape [1, 28, 28].
            ax.imshow(image.squeeze(), cmap="gray")
            ax.set_title(
                f"Predicted: {fashion_classes[prediction]}\n"
                f"Actual: {fashion_classes[label]}"
            )
            ax.axis("off")

    plt.tight_layout()
    plt.show()