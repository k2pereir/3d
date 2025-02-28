import torch
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

#load model 
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(torch.device("cpu"))
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform 

img = cv.imread("penguin.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#monocular depth estimation
input_batch = transform(img).to('cpu')
with torch.no_grad(): 
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), 
        size=img.shape[:2], 
        mode="bicubic",
        align_corners=False,
    ).squeeze()
output = prediction.cpu().numpy()
print(output)
plt.imshow(output)
plt.show()