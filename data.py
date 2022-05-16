from torchvision import datasets, transforms
import config as cf


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image

transform_train_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
]) # meanstd transformation

transform_test_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar10'], cf.std['cifar10']),
])

transform_train_cifar100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
]) # meanstd transformation

transform_test_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
])


def get_dataset(name, train=True, download=True, permutation=None):
    
    if(name == 'cifar10_train'):
        return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_cifar10)
    elif(name == 'cifar10_test'):
        return datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_cifar10)
    elif(name == 'cifar100_train'):
        return datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cifar100)
    elif(name == 'cifar100_test'):
        return datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test_cifar100)
    else:
        print("Invalid dataset mentioned")
        return None

AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10' : datasets.CIFAR10,
    'cifar100' : datasets.CIFAR100
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10' : {'size' : 32, 'channels' : 3, 'classes' : 10},
    'cifar100' : {'size' : 32, 'channels' : 3, 'classes' : 100}
}
