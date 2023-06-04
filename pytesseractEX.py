
#in this programme we will use pytesseract to  extract text from an image  called THE_IMAGE.jpg  after we process it and name it dilated_image


import cv2 
import pytesseract



#before extracting text we have to do some more processing cuz this one contien blocs
gray_image = cv2.imread("gray_image.jpg")
image = cv2.imread("dilated_image.jpg")

bae_image= image.copy()
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #make the image gray

cv2.imwrite('the_gray.jpg',gray)

blur = cv2.GaussianBlur(image,(7,7),0)    #make it flux  
cv2.imwrite("the_blur.jpg", blur)


treash = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cv2.imwrite("the_treshed2.jpg" ,treash)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,13))
cv2.imwrite("index_kernel.jpg",kernel)

dilate=cv2.dilate(treash, kernel, iterations=1)
cv2.imwrite("dilated2.jpg",dilate)

#find th contours
cnts = cv2.findContours(dilate,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts)==2 else cnts[1]

cnts= sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
result=[]
for c in cnts :
    x,y,w,h= cv2.boundingRect(c)   #take the dimontions of contours then i will put nthem in a lisible format of the image 
    if h>20 and h>100 :
        roi= gray_image[y:y+h,x:x+w]     #i used gray image not image cuz gray image is more lisible 
        cv2.rectangle(gray_image,(x,y),(x+w,y+h),(36,255,12),2)   # this line helps to draw a rectangle on the gray_image  the 2 indicates that the rectangle will be drawn with a line thickness of 2 pixels, (36,255,12) indicate the color in this case green (B,G,R) , the other parameters are dimentions
        cv2.imwrite("ROI.jpg",roi)
        cv2.imwrite("rectangle.jpg",gray_image)
        ocr_result = pytesseract.image_to_string(roi)  
        #print(ocr_result)
       



