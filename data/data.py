import torch
import torchvision

def get_train_test_loader(batch_size_train = 64, batch_size_test = 1000, path = './data/'):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
    batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader

