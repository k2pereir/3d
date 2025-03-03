import cv2 as cv 
import torch 
import matplotlib.pyplot as plt 
import numpy as np 
import pyvista as pv 

#load model 
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(torch.device("cpu"))
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

#get 2d image 
img = cv.imread("penguin.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#get depth map 
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

#reshape for 3d point cloud
height, width  = output.shape
y, x = np.indices((height, width))
points = np.stack((x.flatten(), y.flatten(), output.flatten()), axis=-1)
print(points.shape)


print("!!!!!!!!!")
print(output)
print("!!!!!!!!!")

#plt.imshow(output)
#plt.savefig('depthmap.png')

#get 3d point cloud
pc = pv.PolyData(points)
np.allclose(pc.points, points)
pc.plot(eye_dome_lighting=True) #it ends up looking so creepy T_T