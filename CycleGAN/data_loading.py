import os
import glob
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))



class ImageDataset(Dataset):
    def __init__(self, root_path, transforms_=None, unaligned=False, mode = "train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.path_train_a = os.path.join(root_path, "train/A")
        self.path_train_b = os.path.join(root_path, "train/B")
        self.files_A = sorted(glob.glob(self.path_train_a +"/*.*"))
        self.files_B = sorted(glob.glob(self.path_train_b +"/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')


        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    