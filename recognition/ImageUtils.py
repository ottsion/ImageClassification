from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

class MyTestDataSet(DataSet):
    def __init__(self, image_path, transform=None):
        self.data_info = self.get_img_info(image_path)
        self.transform = transform

    def __getitem__(self, index):
        path_img = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, None
    def __le__(self, other):
        return len(self.data_info)

    def get_img_info(self, image_path):
        data_info = []
        for filename in os.listdir(image_path):
            path_img = os.path.join(image_path, filename)
            data_info.append(path_img)
        return data_info

def getTestDataSet(file_path):
    batch_size = 64
    test_data = MyTestDataSet(file_path, transform=image_transforms['test'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    return test_loader


def check_image(path):
    try:
        Image.open(path)
        return True
    except:
        return False

def getDataSet():
    rootpath = '../data'
    batch_size = 128

    image_datasets={name:datasets.ImageFolder(os.path.join(rootpath,name),image_transforms[name], is_valid_file=check_image) for name in ['train','val','test']}
    dataloaders={name : DataLoader(image_datasets[name],batch_size=batch_size,shuffle=True, num_workers=10, pin_memory=True) for name in ['train','val']}
    testDataloader=DataLoader(image_datasets['test'],batch_size=1,shuffle=False, num_workers=10, pin_memory=True)
    print(image_datasets['train'].classes)
    return dataloaders, testDataloader


path = ""