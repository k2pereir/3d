import torch
import cv2 as cv 
import matplotlib.pyplot as plt

#load model 
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(torch.device("cpu"))
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform 

#load camera 
cap = cv.VideoCapture(0)
while True: 
    ret, frame = cap.read()
    if not ret: 
        break; 
    
    #monocular depth estimation 
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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
    plt.imshow(output)
    cv.imshow("frame", frame)
    plt.pause(0.0001)
    
    if cv.waitKey(1) & 0xFF == ord("q"): 
        cap.release()
        plt.show()
        cv.destroyAllWindows()
        break; 
