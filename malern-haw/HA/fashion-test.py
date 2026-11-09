import matplotlib.pyplot as plt
from network import training, evaluate, plot_examples
import torch
import torchvision
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FASHION_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def explorative_data_analysis(train_set, test_set, classes):
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))

    for class_idx in range(10):
        # Find the first 10 examples belonging to this class
        indices = np.where(np.array(train_set.targets) == class_idx)[0][:10]

        for col, idx in enumerate(indices):
            image, label = train_set[idx]

            axes[class_idx, col].imshow(image, cmap="gray")
            axes[class_idx, col].axis("off")

            if col == 0:
                axes[class_idx, col].set_ylabel(
                    classes[class_idx],
                    rotation=0,
                    labelpad=45,
                    va="center"
                )

    plt.suptitle("Fashion-MNIST: Sample Images by Class", fontsize=16)
    plt.tight_layout()
    plt.show()

    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    train_counts = np.bincount(train_labels, minlength=10)
    test_counts = np.bincount(test_labels, minlength=10)

    x = np.arange(len(classes))
    width = 0.4

    plt.figure(figsize=(12, 5))

    plt.bar(x - width / 2, train_counts, width, label="Training")
    plt.bar(x + width / 2, test_counts, width, label="Test")

    plt.xticks(x, classes, rotation=35, ha="right")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution: Training vs. Test")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for class_idx, ax in enumerate(axes.flat):
        class_images = train_set.data[
            train_labels == class_idx
            ].float()

        mean_image = class_images.mean(dim=0)

        ax.imshow(mean_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(classes[class_idx])
        ax.axis("off")

    plt.suptitle("Mean Image per Class", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Convert the training images to floating-point values
    images = train_set.data.float()

    # Mean intensity per image
    mean_intensity = images.mean(dim=(1, 2)).numpy()

    # Distribution of all pixel values
    pixel_values = images.numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(pixel_values, bins=50)
    axes[0].set_xlabel("Pixel intensity")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Pixel Intensities")

    axes[1].hist(mean_intensity, bins=50)
    axes[1].set_xlabel("Mean pixel intensity per image")
    axes[1].set_ylabel("Number of images")
    axes[1].set_title("Distribution of Mean Image Intensities")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    transform = torchvision.transforms.ToTensor()  # converts images to [1,28,28] tensors to use in CNN

    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(fashion_mnist_train, batch_size=64,
                                              shuffle=True, num_workers=2)

    fashion_mnist_test = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(fashion_mnist_test, batch_size=64,
                                             shuffle=False, num_workers=2)

    class_names = fashion_mnist_train.classes

    # For random forest, flattened
    X_train = fashion_mnist_train.data.numpy().reshape(-1, 28 * 28)
    y_train = fashion_mnist_train.targets.numpy()

    X_test = fashion_mnist_test.data.numpy().reshape(-1, 28 * 28)
    y_test = fashion_mnist_test.targets.numpy()

    # explorative_data_analysis(fashion_mnist_train, fashion_mnist_test, class_names)

    print(f"Training samples: {len(fashion_mnist_train)}")
    print(f"Test samples: {len(fashion_mnist_test)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Image shape: {fashion_mnist_train[0][0].size}")
    print(f"Classes: {class_names}")

    forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )
    forest_model.fit(X_train,y_train)

    test_predictions = forest_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, test_predictions):.2%}")
    print(classification_report(
        y_test,
        test_predictions,
        target_names=FASHION_CLASSES,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    cnn_model = training(n_epochs=10, train_set=trainloader)
    metrics = evaluate(testloader, cnn_model, fashion_classes=FASHION_CLASSES)
    print(metrics)
    plot_examples(testloader, FASHION_CLASSES, cnn_model)