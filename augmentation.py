"""
augmentation.py — Data augmentation pipeline for CIFAR100 (student-modified).

Students: Extend the *training* transform pipeline to improve generalization.
The validation pipeline is fixed — do not modify it.

CIFAR100 images are 32×32. Both pipelines resize to 224×224 to match the
input expected by the pretrained ResNet18 backbone.
"""

import torchvision.transforms as T

# Per-channel mean and std computed on the CIFAR100 training set.
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool, train_mode=0) -> T.Compose:
    """Return the image transform pipeline for CIFAR100.

    Args:
        train: If ``True``, returns the training pipeline (with data
               augmentation). If ``False``, returns the validation pipeline
               (deterministic; do not modify).

    Returns:
        A ``torchvision.transforms.Compose`` object ready to be passed to a
        ``torchvision.datasets.CIFAR100`` dataset.

    Student task (training pipeline only):
        The skeleton includes resize, horizontal flip, and normalization.
        Consider adding any of the following to improve generalization:
          - ``T.RandomCrop(224, padding=28)``     — translation invariance
          - ``T.ColorJitter(...)``                — colour robustness
          - ``T.RandomRotation(degrees=15)``      — rotational invariance
          - ``T.RandomErasing(p=0.2)``            — occlusion robustness
          - ``T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)`` — learned policy
        Add transforms *before* ``T.ToTensor()`` (spatial/colour ops) or
        *after* (tensor-level ops such as ``T.RandomErasing``).
    """
    if train:
        if train_mode == 0:      # "full-scale" augmentation
            return T.Compose(
                [
                    T.Resize(224),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                    T.RandomPerspective(distortion_scale=0.2, p=0.3),
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.RandomGrayscale(p=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
                    # ----------------------------------------------------------
                ]
            )
        elif train_mode == 1:      # less severe augmentation
            return T.Compose(
                [  
                    T.Resize(224),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    else:
        # Fixed validation pipeline — do not modify.
        return T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
