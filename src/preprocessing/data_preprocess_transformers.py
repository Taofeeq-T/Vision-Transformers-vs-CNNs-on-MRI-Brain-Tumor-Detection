import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class TumorDataset(Dataset):
    def __init__(self, dataframe, feature_extractor):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        return encoding['pixel_values'].squeeze(0), torch.tensor(label)
