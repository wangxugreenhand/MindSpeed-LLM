from megatron.core.datasets.base_dataset import BaseDataset
from PIL import Image
import torchvision.transforms as transforms
import os

class ViTDataset(BaseDataset):
    def __init__(self, config, data_path, split='train', **kwargs):
        super().__init__(config)
        self.data_path = data_path
        self.split = split
        # 加载数据列表
        self.data = self._load_data()
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_data(self):
        # 加载数据列表，格式：[(image_path, label), ...]
        data = []
        # 实现数据加载逻辑
        # 这里需要根据你的数据集格式进行修改
        # 示例：遍历目录，读取图像和标签
        if os.path.exists(self.data_path):
            for label in os.listdir(self.data_path):
                label_dir = os.path.join(self.data_path, label)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_name)
                        data.append((img_path, int(label)))
        return data
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        # 预处理
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.data)