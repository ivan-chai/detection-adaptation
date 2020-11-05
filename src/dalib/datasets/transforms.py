import PIL

from torchvision import transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1201, 0.1231, 0.1052)


def transform_to_3_channels(x):
    return x.repeat(3, 1, 1)


mnist_transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
    transforms.Lambda(transform_to_3_channels)
])

svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        SVHN_MEAN,
        SVHN_STD
    )
])
