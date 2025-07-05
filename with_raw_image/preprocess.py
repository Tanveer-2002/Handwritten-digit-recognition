import cv2

for i in range(600):
    
    if i<500:
        imgAdd = "RAW_image folder address"+"/"+ str(i) + ".jpg"        #try acutal address for your RAW_image forlder
    else:
        imgAdd = "RAW_image folder address"+"/"+ str(i-500) + ".jpg"

    print(imgAdd)

    image = cv2.imread(imgAdd, cv2.IMREAD_GRAYSCALE)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    #Thresholding (invert if digit is dark on light background)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find Bounding Box around Largest Contour (the digit)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(x,y,w,h)

        # Crop to Bounding Box
        cropped_digit = thresh[y:y+h, x:x+w]

        height, width = cropped_digit.shape
        padding = abs(height - width) // 2

        if height > width:
            padded_digit = cv2.copyMakeBorder(cropped_digit, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
        else:
            padded_digit = cv2.copyMakeBorder(cropped_digit, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

        
        final_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)

        #
        #if i<500: 
        #   cv2.imwrite("trainingImage folder address"+"/"+str(i)+".jpg",final_digit)               #try acutal address for your trainingImage forlder
        #else:
        #   cv2.imwrite("testingImage folder address"+"/"+str(i-500)+".jpg",final_digit             #try acutal address for your testingImage forlder



    else:
        print("No digit found in the image.")
    

    