from dashscope import MultiModalConversation
import json
import os
import dashscope
import dotenv
from dotenv import load_dotenv
from utils import encode_file
from utils import save_image_from_url

class image_editor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALIBABA_API_KEY")
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    
    def edit_image(self, image_filepath, prompt, dest_filename, edit_url):
        encoded_image = encode_file(image_filepath)

        messages = [
        {
            "role": "user",
            "content": [
                {"image": edit_url},
                {"text": prompt}
            ]
        }
        ]
        response = MultiModalConversation.call(
            api_key=self.api_key,
            model="qwen-image-edit",
            messages=messages,
            stream=False,
            watermark=False,
            negative_prompt=" "
        )
        print('got response')

        if response:
            print(response)
            output_url = response.output.choices[0].message.content[0]['image']
            save_image_from_url(output_url, dest_filename)
            return output_url
        else:
            print(f"HTTP status code: {response.status_code}")
            print(f"Error code: {response.code}")
            print(f"Error message: {response.message}")
            print("For more information, see the documentation: https://www.alibabacloud.com/help/zh/model-studio/error-code")

    

