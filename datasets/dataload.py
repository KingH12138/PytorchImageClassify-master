from PIL.Image import open
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class ClassifyDataset(Dataset):
    def __init__(self, image_dir, csv_path, resize):
        super(ClassifyDataset, self).__init__()
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.resize = resize
        self.transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])
        self.df = read_csv(self.csv_path, encoding='utf-8')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = open(self.df['filepath'][idx])
        data = self.transformer(image)
        label = self.df['label'][idx]

        return data, label


def get_dataloader(image_dir, csv_path, resize, batch_size, train_percent=0.9):
    dataset = ClassifyDataset(image_dir, csv_path, resize)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)
