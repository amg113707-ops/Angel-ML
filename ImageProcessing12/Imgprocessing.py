import cv2
import numpy as np
image=cv2.imread("sample.jpg")
#cv2.imshow("Image",image)
grey_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small_image=cv2.resize(grey_image,(3,3))
print("-----")
print(small_image)
#print("Image size:",image)
#print("Grey size:",grey_image)
#print("small size:",small_image)
sobel_x=np.array(
    [
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
    ]
)
sobel_y=np.array(
    [
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
    ]
)
output1=cv2.filter2D(small_image,-1,sobel_x)
output2=cv2.filter2D(small_image,-1,sobel_y)
sobel_combined=cv2.magnitude(output1.astype(np.float32),output2.astype(np.float32))
print(sobel_combined.dtype,sobel_combined.max())
cv2.imshow("Combined Image",cv2.convertScaleAbs(sobel_combined))
cv2.waitKey(0)
cv2.destroyAllWindows()