import os
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

class vlm_analyzer:
    
    def __init__(self):
        load_dotenv()
        category_focus = {
            1: "the tracked object(s) maintaining visual identity (same colors, shapes)",
            2: "the tool/agent staying consistent even as the object transforms",
            3: "the character's appearance (clothing, features) staying identical",
            4: "the character and setting remaining consistent",
            5: "the realism of the images. Limbs, faces, should all be identical",
            6: "the small details, for example: buttons on shirts should be consistent",
            7: "adherence to the prompt"
        }

        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    
        self.prompt = f"""You are a strict consistency checker for sequential images in aphasia research.

        Compare these two images carefully.

        MUST BE IDENTICAL:
        - Character
        - Clothing
        - Setting
        
        MAY CHANGE:
         - Actions, Given prompts. The actions must be consistent with prompts, but other key details remain consistent.
         - Details of the setting/ background / characters, if relevant to the image's respective prompt

        Answer these questions:
        1. Is the SAME character in both images? (yes/no and explain)
        2. Is the clothing identical in color and style? (yes/no and explain)
        3. Is the setting/background the same, and if not, are they accurate with respect to the prompts? (yes/no and explain)
        5. Are the small details that should not change(clothing buttons, etc.) the same? (yes/no and explain)
        6. Does the image adhere to the prompt? (yes/no and explain)
        7. Is this a feasible, real-life scenario (yes/no and explain)

        If ALL answers are YES → respond "PASS:"
        If ANY answer is NO → respond "FAIL:"

        Start your response with either "PASS:" or "FAIL:" followed by your analysis."""

    def check_image_consistency(self, image_1_path, image_1_prompt, image_2_path, image_2_prompt):
        image_1_path = f"file://{image_1_path}"
        image_2_path = f"file://{image_2_path}"

        messages = [{'role':'user',
                # When using a model from the Qwen2.5-VL series with an image list, you can set the fps parameter. This parameter indicates that the images are extracted from a source video at an interval of 1/fps seconds. The setting is ignored for other models.
             'content': [{'image': image_1_path},
                         {'image': image_2_path},
                         {'text': self.prompt + 'image 1 prompt:' + image_1_prompt + 'image 2 prompt' + image_2_prompt}]}]
        
        
        response = MultiModalConversation.call(
        api_key=os.getenv('ALIBABA_API_KEY'),
        model='qwen3-vl-plus',  
        messages=messages)
        
        print(response)
        print(response["output"]["choices"][0]["message"].content[0]["text"])



# vlm = vlm_analyzer()

# vlm.check_image_consistency('raking_6/step_1.png', 'man raking leaves', 'raking_6/step_2.png', 'man bagging leaves')