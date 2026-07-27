from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE , IMAGE_SIZE , TRAIN_DIR,  TEST_DIR


def create_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform , test_transform

def create_dataloaders():

    train_transform , test_transform = create_transforms()

    train_data , test_data , class_names =  create_datasets(train_transform , test_transform )

    train_dataloader = DataLoader(train_data,
                                batch_size=BATCH_SIZE,
                                num_workers=0,
                                    shuffle=True)


    test_dataloader = DataLoader(test_data,
                                 num_workers=0,
                                batch_size=BATCH_SIZE)

    return train_dataloader , test_dataloader , class_names


def create_datasets(train_transform  , test_transform ):

    train_data = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    ) 

    test_data = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transform
    )

    class_names = train_data.classes


    print(f"Training images: {len(train_data)}")
    print(f"Testing images: {len(test_data)}")
    print(f"Classes : {class_names}")
    print(f"Class to Index: {train_data.class_to_idx}")

    return train_data , test_data , class_names






