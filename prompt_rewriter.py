import os 
import dashscope
from dotenv import load_dotenv



class prompt_rewriter:
    def __init__(self):
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        load_dotenv()
        self.api_key = os.getenv("ALIBABA_API_KEY")
        self.edit_system_prompt = """
Given an original prompt and an edit prompt, generate a consistent, succinct prompt to pass to an image edit generation model. Keep all key details the same, but include the desired edits while ensuring to keep other details consistent. 

Crucially, do not add any conversational text, explanations, or warnings. Output only the final, comma-separated, single-paragraph prompt. Make sure to explicitly add the words "NO TEXT" at the end of your response. Do not add any additional characters or details not in the original prompt. Ensure prompts are not > 150 characters.


"""
        self.system_prompt = """
You are an expert AI Image Prompt Engineer. Your sole task is to take a simple, vague, or short image prompt and rewrite it into a consistent, succinct prompt designed to generate a consistent image, newspaper cartoon style with digital coloration. You must add details, but succinct, about the following elements:
    1. Subject: (Age, expression, clothing, pose, action, relationship)
    2. Environment/Setting: (Time of day, weather, location type)
    3. Art Style/Medium: (Newspaper cartoon, hand-drawn, digitally colored, etc.)
    
    Crucially, do not add any conversational text, explanations, additional characters outside the prompt, or warnings. Output only the final, comma-separated, single-paragraph prompt. Make sure to explicitly add the words "NO TEXT" at the end of your response. Do not add any additional characters or details not in the original prompt.
"""

    
    def rewrite_prompt(self, original_prompt):
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': original_prompt}
        ]
        response = dashscope.Generation.call(
        api_key=self.api_key,
        model="qwen-plus",
        messages=messages,
        result_format='message'
        )
        response = response.output.get("choices", [])[0].get("message", {}).get("content", [])
        return(response)
    
    def rewrite_prompt_for_edit(self, original_prompt, edit_prompt):
        messages = [
            {'role': 'system', 'content': self.edit_system_prompt},
            {'role': 'user', 'content': f"Original Prompt: {original_prompt}, Edit Prompt: {edit_prompt}"}
        ]
        response = dashscope.Generation.call(
        api_key=self.api_key,
        model="qwen-plus",
        messages=messages,
        result_format='message'
        )
        response = response.output.get("choices", [])[0].get("message", {}).get("content", [])
        return(response)