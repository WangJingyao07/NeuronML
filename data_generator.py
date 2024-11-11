import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class DataGenerator(Dataset):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.datasource = config.get('datasource', 'sinusoid')
        
        if self.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif 'omniglot' in self.datasource:
            self.num_classes = config.get('num_classes', 5)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            self.metaval_character_folders = character_folders[num_train:num_train+num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        elif self.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', 5)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')

            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label))]
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label))]
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')

    def __len__(self):
        return len(self.metatrain_character_folders) if self.datasource != 'sinusoid' else 200000

    def __getitem__(self, idx):
        if self.datasource == 'sinusoid':
            return self.generate_sinusoid_batch()
        elif 'omniglot' in self.datasource or self.datasource == 'miniimagenet':
            folder = self.metatrain_character_folders[idx % len(self.metatrain_character_folders)]
            rotation = random.choice(self.rotations)
            images, labels = self.get_images(folder, rotation)
            return images, labels
        else:
            raise ValueError('Unrecognized data source')

    def get_images(self, folder, rotation):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(self.img_size),
            transforms.RandomRotation((rotation, rotation)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        images, labels = [], []
        for label, char_folder in enumerate(folder):
            char_images = [os.path.join(char_folder, img) for img in os.listdir(char_folder)]
            sampled_images = random.sample(char_images, self.num_samples_per_class)
            for img_path in sampled_images:
                image = Image.open(img_path)
                image = transform(image)
                images.append(image)
                labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    def generate_sinusoid_batch(self, input_idx=None):
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return torch.tensor(init_inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)


# Example usage
if __name__ == "__main__":
    config = {
        'datasource': 'omniglot',
        'num_classes': 5,
        'img_size': (28, 28),
        'data_folder': './data/omniglot_resized'
    }
    dataset = DataGenerator(num_samples_per_class=5, batch_size=16, config=config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
