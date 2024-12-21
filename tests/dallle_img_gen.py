from openai import OpenAI
import requests 

client = OpenAI()

# response = client.images.create_variation(
#   image=open("/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/alex1.jpg", "rb"),
#   n=1,
#   size="512x512"
# )

response = client.images.generate(
  model="dall-e-2",
  prompt="2D occupancy map.",
  n=1,
  size="256x256"
)
# client.images.edit(
#   image=open("/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/alex2.png", "rb"),
#   prompt="Modify the image to make the person in the image look like jesus. Don't modify the image too much and it should look realistic like a real image.",
#   n=1,
#   size="512x512"
# )

image_url = response.data[0].url
print(response)
print(image_url)

data = requests.get(image_url).content 

f = open("/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/sg.png",'wb') 
f.write(data) 
f.close()