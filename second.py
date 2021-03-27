import face_recognition
import imutils
import pickle
import time
import cv2
import os
import shutil

in_dir = "INPUT/"
out_dir = "OUTPUT/"

# load the known faces and embeddings saved in pickle file
data = pickle.loads(open('face_enc', "rb").read())

#Find path to the input directory you want to detect face 
for im in os.listdir(in_dir):

    image = cv2.imread(in_dir+im)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #convert image to Greyscale for haarcascade (array of True and False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)

    name_predicted = ""

    # loop over the facial embeddings (not necessary)
    for encoding in encodings:

        #Compare encodings with encodings in data["encodings"]
        #Matches contain array with boolean values and True for the embeddings it matches closely
        #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
        encoding)

        #set name =unknown if no encoding matches
        name = "Unknown"
        
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIndex = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIndex:

                #Check the names at respective indexes we stored in matchedIndex
                name = data["names"][i]

                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1

                #set name which has highest count
                name = max(counts, key=counts.get)
    
    
            # the name which appears the most time
            name_predicted = name
           
    # if folder exists already, then copy, else create a folder      
    if os.path.isdir(out_dir+name_predicted):
        shutil.copy(in_dir+im,out_dir+name_predicted+'/'+im)
    else:
        os.makedirs(out_dir+name_predicted+'/')
        shutil.copy(in_dir+im,out_dir+name_predicted+'/'+im)
