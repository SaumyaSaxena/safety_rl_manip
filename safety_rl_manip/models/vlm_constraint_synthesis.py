from enum import Enum
import base64
import os, time
from PIL import Image

from openai import OpenAI
from pydantic import BaseModel, create_model
from safety_rl_manip.envs.utils import add_text_to_img


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def create_response_const_types(object_list, constraint_types, target_choices, use_image=True):

    fields = {}
    for i in range(len(object_list)):
        fields[f"explanation_obj_{object_list[i]}"] = (str, ...)
        fields[f"{object_list[i]}"] = (constraint_types, ...)

    ConstraintTypes = create_model("ConstraintTypes", **fields)

    class ConstraintResponse(BaseModel):
        constraint_types: ConstraintTypes
        target_region: target_choices
        target_description: str
        if use_image:
            image_description: str
    
    return ConstraintResponse

class VLMConstraintSynthesis:
    def __init__(self, vlm_type):
        self.use_image = True
        self.client = OpenAI()
        self._vlm_type = vlm_type
        
    @property
    def agent_role_prompt(self):
        prompt_safety = """
            You are an excellent safe planning agent for dynamic tasks.
            You are given a task description and an image showing the robot and objects on a table.
            The robot is trying to slide the blue box from under the red box and to the right, without damaging other objects along the way.
        """
        return prompt_safety

    def constraint_type_prompt(self, use_image=True):
        prompt = f"""
            Each object on the table can potentially come in contact with the end-effector.
            You need to decide the safe interaction type for each object on the table from the list of constraint types.
            Here the description of the constraint types: 'no_contact' implies that there should absolutely be no contact with a certain object.
            'soft_contact' implies that you can softly interact with that object, push it softly, etc.
            'any_contact' implies that any kind of interaction including aggressive impact is allowed.
            'no_over' implies that the robot is not allowed to move over (on top of) the object.
            Some hints on how to decide on the constraint type for an object:
            If an object is soft or made of durable material, and softly pushing it or moving it without toppling it is okay, 'soft_contact' can be allowed with that object. 
            If an object is very durable, and pushing it aggressively will not damage it, 'any_contact' can be allowed with that object. 
            If an object is fragile, and contacting it might damage it, 'no_contact' should be allowed with that object.
            If an object is very sensitive like an open laptop or a bowl of food, and moving over it might be risky, 'no_over' should be constrained for that object.
            Usually objects such as cups, wine glasses, bowls, electronics, etc are considered fragile and should be 'no_contact'.
            Plastic objects such as bottles, plastic cans, tubes can be allowed 'soft_contact'.
            Soft and non-critical objects such as toys, clothing, etc are soft and can be ignored and allowed 'any_contact'.
            Provide brief explanation, for choosing a specific constraint type for an object. 
            In 'image_description' briefly describe the scene and features relevant to the task.
        """
        return prompt

    @property
    def goal_type_prompt(self):
        prompt = f"""
            You are an excellent safe planning agent for dynamic tasks.
            The robot is trying to slide the blue box from under the red box and place it in one of the two goal regions to 
            the right of the table indicated by the two yellow squares on the table.
            The square closer to the robot is bottom_goal and the square further away is top_goal.
            Choose the target region for the blue box from the two goal regions such that it is safe to slide the blue box to that target region. 
            Also provide a brief explanation for choosing that target region.
        """
        return prompt
    
    def parse_constraint_output(self, output, use_image=True):
        text = ''
        constraints = {}
        # target = output.target_region
        for obj in output.constraint_types:
            if 'explanation' in obj[0]:
                text += obj[0] + ': ' + obj[1] + '\n'
            else:
                text += obj[0] + ': ' + obj[1].value + '\n'
                constraints[obj[0]] = obj[1].value

        if use_image:
            return text, constraints
        else:
            return text, constraints
    
    def get_constraint_types(self, image_path, obj_list, constraint_types, target_choices, use_image=True):

        messages=[
            {"role": "system", "content": f"AGENT ROLE: {self.agent_role_prompt}"},
            {"role": "system", "content": f"Constraint prompt: {self.constraint_type_prompt(use_image=use_image)}"},
            {"role": "system", "content": f"Target/goal prompt: {self.goal_type_prompt}"},
        ]
        if use_image:
            base64_image = encode_image(image_path)
            messages.append(
                { 
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "CURRENT IMAGE: Image showing the objects in the scene."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
            
        enum_constraint_types = Enum('enum_constraint_types', {obj: obj for i, obj in enumerate(constraint_types)}, type=str)
        enum_target_choices = Enum('target_choices', {tar: tar for i, tar in enumerate(target_choices)}, type=str)

        completion = self.client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages,
            response_format=create_response_const_types(obj_list, enum_constraint_types, enum_target_choices, use_image=use_image),
        )
        plan = completion.choices[0].message

        if plan.refusal: # If the model refuses to respond, you will get a refusal message
            return None

        return plan.parsed

from pathlib import Path
if __name__ == "__main__":

    vlm_const_syn = VLMConstraintSynthesis(vlm_type='gpt-4o-2024-08-06')
    constraint_types = ['no_contact', 'soft_contact', 'any_contact', 'no_over']
    object_list = ['salad_bowl', 'porcelain_white_cup', 'supplement_bottle', 'loofah', 'blue_plush_toy', 'blue_die', 'orange_ball', 'xbox_controller']
    target_choices = ['bottom_goal', 'top_goal']
    image_path = Path('/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/vlm_images/')

    for img_file in image_path.glob("*.jpg"):
        print(f"Image: {img_file}")
        start = time.time()
        parsed_plan = vlm_const_syn.get_constraint_types(
            image_path=img_file,
            obj_list=object_list,
            constraint_types=constraint_types,
            target_choices=target_choices,
            use_image=True
        )
        text, constraints = vlm_const_syn.parse_constraint_output(parsed_plan)

        print(f"Time taken: {time.time() - start:.2f} seconds")
        print(f"Constraint Types: {constraints}")
        print(f"target_description: {parsed_plan.target_description}")
        print(f"target_region: {parsed_plan.target_region}")
        import ipdb; ipdb.set_trace()