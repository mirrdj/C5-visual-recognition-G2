from PIL import Image
import os

# Path to the folder containing images
folder_path = '/ghome/group02/C5-G2/Week2/outputs/Detection/faster_rcnn_X_101_32x8d_FPN_3x/final/18'

# Get a list of image filenames in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.gif'))]
print("read image files")

# Sort the image filenames
image_files.sort()
#image_files = image_files[700:850]

print("sorted and cut")

# List to hold the image objects
images = []
print(len(image_files))

# Load each image and append it to the list
for filename in image_files:
    image_path = os.path.join(folder_path, filename)
    img = Image.open(image_path)
    # img = img.resize((1920//4, 1080//4))
    images.append(img)

print("created")

# Save as GIF
images[0].save('Detection_DET_0018.gif',
               save_all=True,
               append_images=images[1:],
               duration=50,  # Change the duration as needed (in milliseconds)
               loop=0)  # 0 means loop forever, change as needed