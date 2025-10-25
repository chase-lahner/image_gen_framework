import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

class DinoEval:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_model = self.dino_model.to(self.device)
        self.dino_model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_image(self, filepath):
       

    
        image = Image.open(filepath).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

   
        with torch.no_grad():
            features = self.dino_model(image_tensor)
         
            

        features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features
    
    def extract_dino_features(self, images):

        image_inputs = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)

        with torch.no_grad():
            image_features = self.dino_model(image_inputs)

            image_features = F.normalize(image_features, p=2, dim=1)

            return image_features
        
    def compute_dino_i(self, images):
        dino_features = self.extract_dino_features(images)

        similarities = []

        for i in range(len(images) - 1):
            sim =(dino_features[i] * dino_features[i+1]).sum().item()
            similarities.append(sim)

        
        avg_similarity = np.mean(similarities)

        return avg_similarity


