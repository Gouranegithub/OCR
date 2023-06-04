import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract

my_image = "THE_IMAGE.jpg"

image = cv2.imread(my_image)
#print the informations of the image!
#print(image)

#to see the image normaly!
#cv2.imshow("the name printed withe the image", image)
#cv2.waitKey(0)

#this function makes u able to display the image like curve! 
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()
    
#display(my_image)

#this hepls u to invert the image wich means white and black!
inverted_image= cv2.bitwise_not( image)

#save the image inverted on jpg format
#cv2.imwrite("inverted.jpg", inverted_image)
#display("inverted.jpg")

def grayscal(the_image):
    return cv2.cvtColor(the_image,cv2.COLOR_RGB2GRAY)

gray_image = grayscal(image)
cv2.imwrite("gray_image.jpg",gray_image)  
#display("gray_image.jpg")


#this helps u to thresh the image (make it more clear)
thresh, im_bw = cv2.threshold(gray_image,210,230,cv2.THRESH_BINARY)
cv2.imwrite("threshed image.jpg",im_bw)

#this function helps u to filter the noise from the image
def nose_removal(image):
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image,kernel,iterations =1)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image,kernel, iterations =1)
    image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    image = cv2.medianBlur(image, 3)
    return image

no_noise = nose_removal(im_bw)
cv2.imwrite("no_noise.jpg",no_noise)

#thise function helps u to erode  the image afre being reverted then revers it again
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel=np.ones((2,2),np.uint8)
    image = cv2.erode(image , kernel, iterations= 1)
    image = cv2.bitwise_not(image)
    return image

eroded_image=thin_font(no_noise)
cv2.imwrite("eroded_image.jpg",eroded_image)

#this function helps u to dilate  the image afre being reverted then revers it again
def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel=np.ones((2,2),np.uint8)
    image = cv2.dilate(image , kernel, iterations= 1)
    image = cv2.bitwise_not(image)
    return image

dilated_image = thick_font(eroded_image)


cv2.imwrite("dilated_image.jpg", dilated_image)



#  ----------------------------the Rotated Image------------------------------------------------------------------------






rotated_image= cv2.imread("rotated_image.jpg")

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

not_rotated = deskew(rotated_image)
cv2.imwrite("not_rotated_any_more.jpg",not_rotated)



#----------------------------------------------------border-------------------------------------------------




#this function helps u to remove border from an image

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)
no_borders = remove_borders(no_noise)
cv2.imwrite("no_borders.jpg", no_borders)


#THIS part helps u to add a border to the image 

color = [225,225,225]         #choose the white color for exp
top, bottom, left, right =[150]*4     #give size of border btw [150]*4= [150,150,150,150]
image_with_borders = cv2.copyMakeBorder(no_borders, top, bottom, left, right,cv2.BORDER_CONSTANT ,value= color)
cv2.imwrite('border_added.jpg',image_with_borders)



#-------------------------------------------pyTesseract OCR------------------------------------


image = cv2.imread("no_noise.jpg")   # we use the the image after removing the noise here where we see the importance of the preprocessing
text = pytesseract.image_to_string(image)
print(text)





