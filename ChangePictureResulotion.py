from PIL import Image
print("start change picture resulotion")
# Open the image
im = Image.open("D:/vegetable-classifier/images_to_predict/0000000004060.jpg")

# Resize the image
im_resized = im.resize((224, 224))

# Save the resized image
im_resized.save("D:/vegetable-classifier/images_to_predict/0000000004060.jpg")