import cv2 as cv 
import torch 
import matplotlib.pyplot as plt 
import open3d 
import numpy as np 

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

print("!!!!!!!!!")
print(output)
print("!!!!!!!!!")

plt.imshow(output)
plt.savefig('depthmap.png')

#get 3d point cloud
pcd = open3d.t.geometry.PointCloud(
    np.array(output, dtype=np.float32))
print(pcd)