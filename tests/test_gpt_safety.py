from enum import Enum
import base64

from openai import OpenAI
from pydantic import BaseModel

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# class TabletopObjects(str, Enum):
#     laptop = "laptop"
#     brown_mug = "brown mug"
#     white_cup = "white cup"
#     pen = "pen"
#     coaster = "coaster"
#     monitor = "monitor"
#     notebook = "notebook"
#     desk = "desk"
#     mouse_pad = "mouse_pad"
#     book = "book"
#     wallet = "wallet"
#     soft_toy = "soft_toy"
#     brown_box = "brown_box"
#     white_box = "white_box"
#     wine_glass = "wine_glass"


class TabletopObjects(str, Enum):
    soft_toy = "soft_toy"
    purple_box = "purple_box"
    yellow_box = "yellow_box"
    glass_kettle = "glass_kettle"


def create_planner_response():

    class RelevantObjects(BaseModel):
        explanation_obj1: str
        object1: TabletopObjects
        explanation_obj2: str
        object2: TabletopObjects
        explanation_obj3: str
        object3: TabletopObjects

    class PlannerResponse(BaseModel):
        relevant_objects: RelevantObjects
        image_description: str
    
    return PlannerResponse

class SafePlanner:
    def __init__(self):
        self.use_image = True
        self.client = OpenAI()
        self._vlm_type = "gpt-4o-2024-08-06"
    
    @property
    def agent_role_prompt(self):
        prompt = """
        You are an excellent safe planning agent. Given a task and a sequence of images, where you are required to 
        figure out the safety-critical objects on a cluttered table, 
        you figure out which objects are relevant to the task and to safety and which can be ignored. 
        In order to do that, the two criteria you consider are 'task success' and 'safety'.
        Under the 'task success' criteria you figure out which objects need to be manipulated or interacted with to accomplish the task.
        Under the 'safety' criteria you consider the objects that you need to move carefully around to prevent damage, spillage, collision, or other safety critical criteria.
        You are allowed to select three distinct objects that are relevant for the task and also provide explanation for why you are choosing these objects,
        specifically mention if a selected object is 'task critical' or 'safety critical' or both.
        Also, provide a brief description of the sequence of images, paying attention to the features in the images that are task relevant and safety critical.
        The image contains a set of sequential timeshots taken in the scene, which depict the motion of objects in the recent past.
        """
        return prompt
    
    @property
    def task_prompt(self):
        # prompt = """
        #     Pick up the brown coffee mug and place it next to the white notepad on the right. Be careful, the brown coffee mug is full of hot coffee.
        # """
        prompt = """
            Slide the purple box from under the yellow box and move towards the edge of the table where some other object is placed.
        """
        return prompt
    
    def get_gpt_output(self, image_path):
        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"TASK: {self.task_prompt}"},
        ]
        if self.use_image:
            base64_image = encode_image(image_path)
            messages.append(
                { 
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "CURRENT IMAGE: This image represents the current view of the agent."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
        
        completion = self.client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages,
            response_format=create_planner_response(),
        )
        plan = completion.choices[0].message

        if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
            return plan

        return plan

from PIL import Image, ImageDraw, ImageFont
import pillow_heif
from pathlib import Path
import os

def save_png_from_heic(img_path):
    heic_files = [f for f in os.listdir(img_path) if f.lower().endswith(".heic")]
    for i, heic_file in enumerate(heic_files):
        file = pillow_heif.open_heif(img_path+heic_file)
        image = Image.frombytes(
            file.mode, 
            file.size, 
            file.data, 
            "raw"
        )
        scale = 0.1  # Scale down to 50%
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        low_res_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        low_res_image.save(img_path+f"img_low_res_{i}.png", "PNG")

def append_and_save_png(img_path, png_files):
    images = [Image.open(img_path+png_file) for png_file in png_files]

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a blank canvas
    combined_image = Image.new("RGBA", (total_width, max_height))

    # Paste images onto the canvas
    x_offset = 0
    for i, img in enumerate(images):

        draw = ImageDraw.Draw(img)
        text = f"Image {i}"
        position = (10, 10)  # Top-left corner for text
        text_color = (255, 255, 255)  # White color (R, G, B)
        # Add the text to the image
        draw.text(position, text, fill=text_color, font=ImageFont.load_default())
        # draw.text(position, text, fill=text_color, font=ImageFont.truetype("arial.ttf", 40))

        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_image.save(img_path+"full_traj_franka_teapot_teddy.png", "PNG")

import cv2
def mov_to_pngs(mov_path):
    output_folder = os.path.dirname(mov_path)
    cap = cv2.VideoCapture(mov_path)

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    
    scale_factor = 0.25
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count < 100:
            frame_count += 1
            continue

        if frame_count % 5 == 0:
            original_height, original_width = frame.shape[:2]
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            new_resolution = (new_width, new_height)

            resized_frame = cv2.resize(frame, new_resolution)
            # Save each frame as a PNG
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, resized_frame)
        frame_count += 1
    
    cap.release()
    print(f"Saved {frame_count} frames to {output_folder}")
    
if __name__ == "__main__":

    # mov_to_pngs("/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/gpt_safety/slide_pickup/slide_pick_obs/slide_pick_obs.MOV")

    planner = SafePlanner()

    # img_path = '/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/gpt_safety/imgs_coffee/move_over_laptop/'
    img_path = '/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/slide_pickup_clutter/'
    # save_png_from_heic(img_path)
    # png_files = ["img_low_res_1.png", "img_low_res_0.png", "img_low_res_3.png", "img_low_res_2.png"]
    # png_files = ["img_low_res_2.png", "img_low_res_1.png", "img_low_res_0.png"]
    # png_files = ["frame_0290.png", "frame_0305.png", "frame_0310.png", "frame_0325.png"]
    # png_files = ["frame_0700.png", "frame_0705.png", "frame_0710.png", "frame_0720.png"]
    png_files = ["rgb_slide_pickup_clutter_t_152_0_front.png", "rgb_slide_pickup_clutter_t_164_0_front.png", "rgb_slide_pickup_clutter_t_180_0_front.png", "rgb_slide_pickup_clutter_t_190_0_front.png"]
    append_and_save_png(img_path, png_files)
    import ipdb; ipdb.set_trace()

    png_files = [f for f in os.listdir(img_path) if (f.lower().endswith(".png") and 'full_traj_franka_teapot_teddy' in f.lower())]
    for img in png_files:
        gpt_output = planner.get_gpt_output(img_path+img)
        print(gpt_output.parsed)
        import ipdb; ipdb.set_trace()