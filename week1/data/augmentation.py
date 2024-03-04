import torchvision.transforms as transforms

def data_augmentation(augment: bool, params, width, height):
    """
    Generates augmented data.
    :param augment: Boolean, when true the data augmentation is done.
    :return: ImageDataGenerator object instance
    """
    rotation_range=0
    width_shift_range=0.
    height_shift_range=0.
    shear_range=0.
    zoom_range=0.
    horizontal_flip=0


    if 'rotation' in params['data_aug']:
        rotation_range = 20
    if 'wsr' in params['data_aug']:
        width_shift_range=0.2
    if 'hsr' in params['data_aug']:
        height_shift_range=0.2
    if 'sr' in params['data_aug']:
        shear_range=0.2
    if 'zr' in params['data_aug']:
        zoom_range=0.2
    if 'hf' in params['data_aug']:
        horizontal_flip=0.2

    if augment:
        transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.RandomHorizontalFlip(horizontal_flip),
        transforms.RandomRotation(rotation_range),
        transforms.RandomAffine(degrees=0, shear=(-shear_range, shear_range)), #Shear
        transforms.RandomResizedCrop(size=(width, height), scale=(1.0-zoom_range, 1.0)), #Zoom
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.43239, 0.45149, 0.44213],
            std=[0.255, 0.244, 0.270]
        )])
            
        return transform

    else:
        transform = transforms.Compose([
                    transforms.Resize((width, height)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.43239, 0.45149, 0.44213],
                    std=[0.255, 0.244, 0.270]
                    )])
        
        return transform