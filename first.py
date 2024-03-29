from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
#get paths of each file in folder named data
#Different folders are created to hold different persons images separately
imagePaths = list(paths.list_images('data'))
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):

    # extract the person name from the image path(-2 index)
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "labels": knownNames}

#use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()