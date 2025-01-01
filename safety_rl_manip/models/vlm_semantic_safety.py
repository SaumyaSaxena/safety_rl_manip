from enum import Enum
import base64

from openai import OpenAI
from pydantic import BaseModel

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# class TabletopObjects(str, Enum):
#     soft_toy = "soft_toy"
#     glass_kettle = "glass_kettle"



def create_planner_response(object_list):
    class RelevantObjects(BaseModel):
        explanation_obj1: str
        object1: object_list
        # explanation_obj2: str
        # object2: object_list
        # explanation_obj3: str
        # object3: object_list

    class PlannerResponse(BaseModel):
        relevant_objects: RelevantObjects
        image_description: str
    
    return PlannerResponse

class SafePlanner:
    def __init__(self, vlm_type, all_objects):
        self.use_image = True
        self.client = OpenAI()
        self._vlm_type = vlm_type
        self.all_objects = all_objects
        self.object_list = Enum('object_list', {obj: obj for obj in all_objects}, type=str)
    
    @property
    def agent_role_prompt(self):
        prompt = """
        You are an excellent safe planning agent for dynamic tasks.
        Ypu are given a task description and a sequence of images showing the trajectory followed by the robot and the objects so far, 
        You are required to figure out the safety-critical objects on a cluttered table. 
        Under the 'safety' criteria you consider the objects that you need to move carefully around to prevent damage, spillage, collision, or other safety critical criteria.
        Provide explanation for why you are choosing these objects and why they are safety critical given the current trajectory. 
        Pay special attention to the robot's trajectory in the sequence of images for choosing objects that can make most probable contacts.
        Also, provide a brief description of the sequence of images, paying attention to the features in the images that are task relevant and safety critical.
        """
        return prompt
    
    @property
    def task_prompt(self):
        prompt = """
            A robot arm needs to slide the purple box from under the yellow box and move towards the edge of the table. 
            Other objects are placed on the table and the robot needs to choose which ones to avoid.
        """
        return prompt

    def get_text_from_parsed_output(self, output):
        text = ''
        objs = []
        for obj in output.relevant_objects:
            if 'object' in obj[0]:
                text += obj[0] + ': ' + obj[1].value + '\n'
                objs.append(obj[1].value)
            else:
                text += obj[0] + ': ' + obj[1] + '\n'
        return text, objs
    
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
                        "text": "CURRENT IMAGE: Sequence of images showing the trajectory followed by the robot and the objects so far."
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
            response_format=create_planner_response(self.object_list),
        )
        plan = completion.choices[0].message

        if plan.refusal: # If the model refuses to respond, you will get a refusal message
            return None

        return plan.parsed