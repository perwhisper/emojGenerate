from torchvision import transforms

def get_augmentation_pipeline(role_type="real", img_size=256):
    base_aug = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    if role_type == "cartoon":
        base_aug.insert(3, transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
    elif role_type == "handdrawn":
        base_aug.insert(3, transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)))
    return transforms.Compose(base_aug)
