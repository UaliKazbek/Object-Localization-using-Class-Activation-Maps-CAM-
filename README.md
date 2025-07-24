# Object-Localization-using-Class-Activation-Maps-CAM-
This script performs object localization using ResNet34 and Class Activation Maps (CAM).
## Features
- Uses pretrained ResNet34 from torchvision
- Extracts intermediate feature maps
- Computes weighted sum using final FC layer weights
- Generates activation map and bounding box
- Identifies and returns the predicted class
## Sample Input and Output
``` text
img, cam_map, bbox, class_name = localization("path/to/image.jpg")

print(class_name)   # e.g. 'Egyptian cat'
print(bbox)         # e.g. (x1, y1, x2, y2)

plt.imshow(cam_map, cmap='gray')
plt.show()

plt.imshow(img)
plt.show()
```
