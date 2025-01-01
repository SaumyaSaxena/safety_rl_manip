from enum import Enum
import base64

from openai import OpenAI
from pydantic import BaseModel
from safety_rl_manip.utils import save_png_from_heic, append_and_save_png, mov_to_pngs

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
    append_and_save_png(img_path, png_files, 'full_traj_franka_teapot_teddy')
    import ipdb; ipdb.set_trace()

    png_files = [f for f in os.listdir(img_path) if (f.lower().endswith(".png") and 'full_traj_franka_teapot_teddy' in f.lower())]
    for img in png_files:
        gpt_output = planner.get_gpt_output(img_path+img)
        print(gpt_output.parsed)
        import ipdb; ipdb.set_trace()