from enum import Enum
import base64
import os
from PIL import Image

from openai import OpenAI
from pydantic import BaseModel
from safety_rl_manip.envs.utils import add_text_to_img

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# class TabletopObjects(str, Enum):
#     soft_toy = "soft_toy"
#     glass_kettle = "glass_kettle"

def create_planner_response(object_list, constraint_types_list):
    class RelevantObjects(BaseModel):
        explanation_obj1: str
        object1: object_list
        constraint_type_object1: constraint_types_list
        explanation_obj2: str
        object2: object_list
        constraint_type_object2: constraint_types_list
        explanation_obj3: str
        object3: object_list
        constraint_type_object3: constraint_types_list

    class PlannerResponse(BaseModel):
        relevant_objects: RelevantObjects
        image_description: str
    
    return PlannerResponse

class VLMSafetyCriteria:
    def __init__(self, vlm_type, constraint_types):
        self.use_image = True
        self.client = OpenAI()
        self._vlm_type = vlm_type
        self.constraint_types = constraint_types
    
    @property
    def agent_role_prompt(self):
        prompt_safety = """
            You are an excellent safe planning agent for dynamic tasks.
            You are given a task description and a sequence of images showing the trajectory followed by the robot and the objects so far,
            the last image shows the current state.
            The objects are highlighed using bounding boxes with labels <object id>: <object name>.
            The past trajectory of the robot can be inferred from the sequence of images and also from the black arrow showing the past path followed by the robot end effector. 
            Important note: the black line with the arrow shows the past trajectory already followed by the end-effector in the past few time steps, not the planned future trajectory.
            Expecting that the robot will continue along the path, try to predict the future trajectory and
            choose relevant objects in the future: You need to do this based on two criteria: proximity and safety.
            Under the 'proximity' criteria you consider the objects which are most likely to come in contact with the top cereal box or end-effector if the robot continues along its its trajectory in the future.
            The object list provided to you is sorted in increasing order of the distance between the end-effector and the object. 
            For example, object with index 0 will be closest to the end-effector, object with index 1 will be further away and so on. Use this list to help with the proximity criteria.
            Under the 'safety' criteria, among the objects that might come in contact in the near future, choose objects which are more fragile/safety critical, and coming in contact with them is more risky.
            Provide explanation for why you are choosing these objects (proximity vs safety) and why they are relevant in the near future. 
            In 'image_description' briefly describe 1) brief description of the sequence of images, 2) describe the trajectory followed by the robot (black arrow) so far 3) describe what the future trajectory will look like 4) which objects might likely come in contact with the gripper in the future.
        """
        return prompt_safety
    
    @property
    def task_prompt(self):
        prompt = """
            A robot arm needs to slide the white cereal box from under the blue cereal box and to the right, carefully without hitting any other objects along the way. 
            Other objects are placed on the table and the agent needs to choose which ones to avoid.
        """
        return prompt

    def get_text_from_parsed_output(self, output):
        # print(output.image_description)
        text = ''
        objs, obj_const_types = [], []
        for obj in output.relevant_objects:
            if 'constraint_type_object' in obj[0]:
                text += obj[0] + ': ' + obj[1].value + '\n'
                obj_const_types.append(obj[1].value)
            elif 'object' in obj[0]:
                text += obj[0] + ': ' + obj[1].value + '\n'
                objs.append(obj[1].value)
            else:
                text += obj[0] + ': ' + obj[1] + '\n'
        return text, output.image_description, objs, obj_const_types
    
    def get_gpt_output(self, image_path, sorted_obj_names):
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
                        "text": "CURRENT IMAGE: Sequence of images showing the trajectory followed by the robot and the objects so far."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
        
        sorted_object_list = Enum('sorted_object_list', {str(i): obj for i, obj in enumerate(sorted_obj_names)}, type=str)
        constraint_types_list = Enum('constraint_types_list', {str(i): obj for i, obj in enumerate(self.constraint_types)}, type=str)

        completion = self.client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages,
            response_format=create_planner_response(sorted_object_list, constraint_types_list),
        )
        plan = completion.choices[0].message

        if plan.refusal: # If the model refuses to respond, you will get a refusal message
            return None

        return plan.parsed

    def get_rel_objects(self, image_path, sorted_obj_names):
        plan_parsed = self.get_gpt_output(image_path, sorted_obj_names)
        obj_text, img_desc, objs, obj_const_types = self.get_text_from_parsed_output(plan_parsed)

        img_vlm_out = add_text_to_img(image_path, obj_text)
        directory, filename = os.path.split(image_path)
        base, ext = os.path.splitext(filename)
        Image.fromarray(img_vlm_out).save(os.path.join(directory, f"{base}_vlm_output{ext}"))

        return objs, obj_const_types, obj_text, img_desc