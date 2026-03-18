import os
from PIL import Image

input_dir = "/root/autodl-fs/datasets/validation_camera_poses/WestTeachingAreas/images/LR"
output_dir = "/root/autodl-fs/HR_bicubic/WestTeachingAreas"

os.makedirs(output_dir, exist_ok=True)

count = 0

for name in os.listdir(input_dir):

    if name.lower().endswith((".png",".jpg",".jpeg")):

        path = os.path.join(input_dir, name)

        try:
            img = Image.open(path)

            hr = img.resize((1368, 912), Image.BICUBIC)

            hr.save(os.path.join(output_dir, name))

            count += 1

        except Exception as e:
            print("Error:", name, e)

print("Saved images:", count)