import torch
import torchvision

def get_train_test_loader(dataset_name='mnist', batch_size_train = 64, batch_size_test = 1000, path = './data/'):
    transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])
    supported_datasets = ['mnist','fashionmnist']
    if dataset_name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(path, train=True, download=True,
                                        transform=transform),
        batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(path, train=False, download=True,
                                        transform=transform),
        batch_size=batch_size_test, shuffle=True)
    elif dataset_name == 'fashionmnist':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(path, train=True, download=True,
                                        transform=transform),
        batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(path, train=False, download=True,
                                        transform=transform),
        batch_size=batch_size_test, shuffle=True)

    else:
        raise NotImplemented(f"Dataset: {dataset_name} not implemented yet. Please supply one of the following as dataset_name, {supported_datasets}")
    
    return train_loader, test_loader

