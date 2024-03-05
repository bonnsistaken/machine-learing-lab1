import cv2

def apply_gabor_filter(image_path, frequency, theta):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getGaborKernel((21, 21), sigma=5.0, theta=theta, lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    ima="C:/Users/Pc/Desktop/padugeevitham/ravi.png"
    cv2.imshow('Filtered Image', filtered_image)
    cv2.imwrite(ima,filtered_image)
    f=filtered_image.flatten()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "C:/Users/Pc/Pictures/Screenshots/Screenshot (28).png"
apply_gabor_filter(image_path, 0.7, 0.5)
apply_gabor_filter(image_path,0.65,0.45)
apply_gabor_filter(image_path,0.675,0.475)
apply_gabor_filter(image_path,0.68,0.48)

        


