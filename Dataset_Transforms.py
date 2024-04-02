import matplotlib.pyplot as plt
import torchvision
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

ds_mean = (0.4914, 0.4822, 0.4465)
ds_std = (0.247, 0.243, 0.261)

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(
        self, root="../data/cifar10", train=True, download=True, transform=None
    ):
        super().__init__(root=root, train=train, download=download, transform=transform)

        if transform == "train":
            self.transform = A.Compose([
                A.Normalize(
                    mean=ds_mean, std=ds_std
                  ),
                A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
              # A.PadIfNeeded(min_height=32+4, min_width=32+4, position='center', border_mode=0, value=ds_mean, mask_value=ds_mean),
                A.RandomCrop(32, 32),
                A.HorizontalFlip(),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=8,
                    max_width=8,
                    fill_value=ds_mean,
                    mask_fill_value = None,
                ),
                ToTensorV2(),
            ]
        )
        elif transform == "test":
            self.transform = A.Compose([
                A.Normalize(mean=ds_mean, std=ds_std),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
		
# train_transforms = A.Compose(
            # [
                # A.Normalize(
                    # mean=ds_mean, std=ds_std
                  # ),
                # A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
              # # A.PadIfNeeded(min_height=32+4, min_width=32+4, position='center', border_mode=0, value=ds_mean, mask_value=ds_mean),
                # A.RandomCrop(32, 32),
                # A.HorizontalFlip(),
                # A.CoarseDropout(
                    # max_holes=1,
                    # max_height=8,
                    # max_width=8,
                    # fill_value=ds_mean,
                    # mask_fill_value = None,
                # ),
                # ToTensorV2(),
            # ]
        # )

# # Test data transformations
# test_transforms = A.Compose([
                # A.Normalize(
                    # mean=ds_mean, std=ds_std
                # ),
                # ToTensorV2(),
    # ])
	
