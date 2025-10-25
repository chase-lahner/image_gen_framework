import os
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation
import json
from utils import save_image_from_url





class image_generator:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALIBABA_API_KEY")
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

    def generate_image(self, prompt, filename):
        messages = [
        {
            "role": "user",
            "content" : [
                {"text": prompt}
            ]
        }
        ]
        response = MultiModalConversation.call(
        api_key=self.api_key,
        model="qwen-image-plus",
        messages=messages,
        result_format='message',
        stream=False,
        watermark=False,
        prompt_extend=True,
        negative_prompt='',
        size='1328*1328'
        )

        if response.status_code == 200:
            print(json.dumps(response, ensure_ascii=False))
        else:
            print(f"HTTP status code: {response.status_code}")
            print(f"Error code: {response.code}")
            print(f"Error message: {response.message}")
            print("For more information, see the documentation: https://www.alibabacloud.com/help/en/model-studio/error-code")

        try:
            content = response.output.get("choices", [])[0].get("message", {}).get("content", [])
            image_url = next((item["image"] for item in content if isinstance(item, dict) and "image" in item), None)
            if not image_url:
                raise KeyError("No image field found in response.")
            print(f"Image URL: {image_url}")
            save_image_from_url(image_url, filename)
            return image_url
        except (IndexError, KeyError, AttributeError, TypeError) as e:
            print(f"Failed to extract image URL: {e}")


# Need: to initialize functions that -> rewrite prompts _
    # qwen 2.5 - 7b
    # need to rewrite prompts
# Need: to initalize functions that -> vlm initialize
    # evaluates image
    #  creates prompt from image
# Need: to create evaluation framework -> CLIP / DINO / VLM evaluation ^

# workflow
# 1) prompt user to input # steps in their sequence
# 2) create first prompt -> vintage, cartoon style, no text
# 3) give to user for validation
# 4) give generated image
# 5) give to user for validation
# 6) given image and second sequence, give enhanced prompt to user
# 7) generate enhanced prompt using qwen-image-edit



