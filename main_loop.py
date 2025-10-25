from image_gen import image_generator
from image_edit import image_editor
from prompt_rewriter import prompt_rewriter
import os
from utils import encode_file
from PIL import Image
from dino_eval import DinoEval
from clip_eval import CLIPEvaluator
from vlm_analyzer import vlm_analyzer


def main():
    """
    Interactive workflow for generating sequential images with consistent style.
    """
    print("="*70)
    print("INTERACTIVE IMAGE SEQUENCE GENERATOR")
    print("="*70)
    

    generator = image_generator()
    editor = image_editor()
    rewriter = prompt_rewriter()
    evaluator = DinoEval()
    clip_evaluator = CLIPEvaluator()
    vlm = vlm_analyzer()
    
    output_dir = input("specify output directory: ")
    os.makedirs(output_dir, exist_ok=True)
    
    
    print("\n[STEP 1] Setup")
    print("-"*70)
    num_steps = int(input("How many steps in your sequence? "))
    print(f"You will create {num_steps} images.\n")

    goal_step = input("input goal: ")
    
    
    images = []
    prompts = []
    edit_prompts = []
    pil_images = []
    
    
    print("\n[STEP 2] First Prompt")
    print("-"*70)
    user_prompt = input("Enter your first image description: ")
    
    
    print("\nRewriting prompt with vintage cartoon style...")
    enhanced_prompt = rewriter.rewrite_prompt(user_prompt)
    
   
    print("\n[STEP 3] Prompt Validation")
    print("-"*70)
    print("Enhanced prompt:")
    print(f"\n{enhanced_prompt}\n")
    proceed = input("Generate image with this prompt? (y/n): ").strip().lower()
    
    if proceed != 'y':
        enhanced_prompt = input("Please input propt to pass to generator")
    
    prompts.append(enhanced_prompt)
    
    
    print("\n[STEP 4] Generating Image")
    print("-"*70)
    first_image = f"{output_dir}/step_1.png"
    print(f"Generating image 1/{num_steps}...")
    edit_url = generator.generate_image(enhanced_prompt, first_image)
    images.append(first_image)
    image = Image.open(first_image)
    image.show()
    pil_images.append(Image.open(first_image))
    
    print("\n[STEP 5] Image Validation")
    print("-"*70)
    print(f"Image saved to: {first_image}")
    proceed = input("Does the image look good? Continue to next step? (y/n): ").strip().lower()
    
    while proceed != 'y':
        edit = input("Would you like to regenerate this image with edits? (y/n)").strip().lower()
        if edit != 'y':
            print('exiting')
            return
        edit_prompt = input("Input prompt to edit image")
        edit_prompts.append(edit_prompt)
        editor.edit_image(first_image, edit_prompt, first_image, edit_url)
        image = Image.open(first_image)
        pil_images[0] = Image.open(first_image)
        image.show()
        proceed = input("does this image look good? Continue to next step (y/n)")

    prev_prompt = enhanced_prompt
    
    
    for step_num in range(2, num_steps + 1):
        print("\n" + "="*70)
        print(f"[STEP {step_num}] Next Image in Sequence")
        print("="*70)
        
 
        prev_image = images[-1]
        user_prompt = input(f"\nDescribe what happens in step {step_num}: ")
        
        # # Rewrite prompt
        # print("\nRewriting prompt...")
        # enhanced_prompt = rewriter.rewrite_prompt_for_edit(prev_prompt, user_prompt)
        
        # print("\nEnhanced prompt:")
        # print(f"\n{enhanced_prompt}\n")
        enhanced_prompt = user_prompt
        proceed = input("Enhance this prompt? (y/n): ").strip().lower()
        
        if proceed != 'y':
            enhanced_prompt = rewriter.rewrite_prompt(enhanced_prompt)
        
        prompts.append(enhanced_prompt)

        print(f"\nGenerating image {step_num}/{num_steps} based on previous image...")
        next_image = f"{output_dir}/step_{step_num}.png"
        prev_url = edit_url
        cur_url = editor.edit_image(prev_image, enhanced_prompt, next_image, edit_url)
        images.append(next_image)
        
        
        print(f"\nImage saved to: {next_image}")
        image = Image.open(next_image)
        image.show()
        pil_images.append(image)
        proceed = input("Continue to next step? (y/n): ").strip().lower()
        # print("VLM evaluation -----------------------------------------")
        # vlm.check_image_consistency(images[-2], prompts[-2], images[-1], prompts[-1])
        
        while proceed != 'y':
            edit = input("Would you like to regenerate this image with edits? (y/n)").strip().lower()
            if edit != 'y':
                print("now editing previous one")
                edit_prmpt = input("input prompt to regenerate image")
                edit_prompts.append(edit_prompt)
                cur_url = editor.edit_image(image, edit_prmpt, next_image, cur_url)
                image = Image.open(next_image)
                image.show()
                pil_images[-1] = image
                proceed = input("does this image look good (y/n)?")
            if edit == 'y':
                edit_prompt = input("Input prompt to regenerate image")
                edit_prompts.append(edit_prompt)
                cur_url = editor.edit_image(prev_image, edit_prompt, next_image, prev_url)
                image = Image.open(next_image)
                image.show()
                pil_images[-1] = image
                proceed = input("does this image look good? Continue to next step (y/n)")
    

    print("\n" + "="*70)
    print("SEQUENCE COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(images)} images:")
    for i, img_path in enumerate(images, 1):
        print(f"  Step {i}: {img_path}")
    
    print(f"\nAll prompts used:")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  Step {i}:")
        print(f"  {prompt}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

    print("evaluating images")
    dino_i = evaluator.compute_dino_i(pil_images)
    clip_t = clip_evaluator.compute_clip_t(pil_images, prompts)
    clip_star = clip_evaluator.compute_clip_star(pil_images, prompts)

    dino_star = dino_i * clip_t
    print("DINO-i", dino_i)
    print("CLIP-star", clip_star)
    print("DINO-star", dino_star)

    goal_faithfulness = clip_evaluator.compute_goal_faithfulness(pil_images, goal_step, prompts, 1)

    print("goal faithfulness ", goal_faithfulness)


if __name__ == "__main__":
    main()
