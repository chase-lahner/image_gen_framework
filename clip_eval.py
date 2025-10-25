from transformers import CLIPProcessor, CLIPModel
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn.functional as F
import random

class CLIPEvaluator:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()
        self.device = 'cpu'

    def extract_clip_image_features(self, images):
        inputs = self.clip_processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def extract_clip_text_features(self, texts):
        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def compute_clip_i(self, images):
        """
        CLIP-I: Average cosine similarity between consecutive images
        """
        if len(images) < 2:
            return 0.0
        
        clip_features = self.extract_clip_image_features(images)
        
      
        similarities = []
        for i in range(len(images) - 1):
            sim = (clip_features[i] * clip_features[i + 1]).sum().item()
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        return avg_similarity
    
    def compute_clip_t(self, images, 
                       texts):
        """
        CLIP-T: Text-image alignment
        For each step, compare image with average of all text up to that step
        """
        if len(images) != len(texts):
            raise ValueError("Number of images must match number of texts")
        
        clip_image_features = self.extract_clip_image_features(images)
        clip_text_features = self.extract_clip_text_features(texts)
        
      
        alignments = []
        for i in range(len(images)):
           
            cumulative_text = clip_text_features[:i+1].mean(dim=0, keepdim=True)
            cumulative_text = F.normalize(cumulative_text, p=2, dim=1)
            
         
            sim = (clip_image_features[i:i+1] * cumulative_text).sum().item()
            alignments.append(sim)
        
        avg_alignment = np.mean(alignments)
        
        return avg_alignment
    
    def compute_clip_star(self, images, 
                          texts):
        """
        CLIP*: Product of CLIP-I and CLIP-T
        """
       
        clip_i = self.compute_clip_i(images)
        clip_t = self.compute_clip_t(images, texts)
        print("clip i, t", clip_i, clip_t)
        return clip_i * clip_t
    
        
    def compute_goal_faithfulness(self, images, goal_text, all_goal_texts, num_distractors=3):
        """
        Goal Faithfulness: Multiple-choice accuracy
        For each image, check if correct goal has highest CLIP similarity vs distractors
        
        Args:
            images: List of PIL Images for ONE sequence
            goal_text: The correct goal text for this sequence
            all_goal_texts: List of ALL goal texts in dataset (for sampling distractors)
            num_distractors: Number of distractor goals (default 3)
        
        Returns:
            Accuracy (0.0 to 1.0)
        """
        if len(images) == 0:
            return 0.0
        
        
        image_features = self.extract_clip_image_features(images)
        
        correct_count = 0
        total_count = 0
        
        for img_feat in image_features:
            
            available_distractors = [g for g in all_goal_texts if g != goal_text]
            
            if len(available_distractors) < num_distractors:
              
                continue
            
            distractors = random.sample(available_distractors, num_distractors)
            
          
            candidates = [goal_text] + distractors
            
        
            text_features = self.extract_clip_text_features(candidates)
            
            
            similarities = (img_feat.unsqueeze(0) @ text_features.T).squeeze(0)
            
           
            predicted_idx = similarities.argmax().item()
            if predicted_idx == 0:
                correct_count += 1
            total_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return accuracy

    def compute_step_faithfulness(self, images, step_texts, goal_text):
        """
        Step Faithfulness: Multiple-choice accuracy
        For each image, check if its corresponding step text has highest similarity
        among all steps in the same goal
        
        Args:
            images: List of PIL Images for ONE sequence
            step_texts: List of step texts corresponding to images
            goal_text: The goal text (used for conditioning as in paper)
        
        Returns:
            Accuracy (0.0 to 1.0)
        """
        if len(images) != len(step_texts) or len(images) == 0:
            print(len(images))
            print("ERRRRRRR")
            return 0.0
        
      
        image_features = self.extract_clip_image_features(images)
        
        
        conditioned_steps = [f"{goal_text}. {step}" for step in step_texts]
        
     
        text_features = self.extract_clip_text_features(conditioned_steps)
        
        correct_count = 0
        total_count = 0
        
        for step_idx, img_feat in enumerate(image_features):
           
            similarities = (img_feat.unsqueeze(0) @ text_features.T).squeeze(0)
            
           
            predicted_idx = similarities.argmax().item()
            if predicted_idx == step_idx:
                correct_count += 1
            total_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return accuracy
    

