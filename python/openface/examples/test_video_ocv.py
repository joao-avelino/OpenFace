#!/u sr/bin/python3
import sys
import cv2 as cv
import numpy as np
from openface import pyopenface as of

cap = cv.VideoCapture("crop_impaciente.avi")

#Initialize openface

det_parameters = of.FaceModelParameters()

det_parameters.model_location = "/home/sims/repositories/OpenFace/build/bin/model/main_ceclm_general.txt"

det_parameters.mtcnn_face_detector_location = "/home/sims/repositories/OpenFace/build/bin/model/mtcnn_detector/MTCNN_detector.txt"

face_model = of.CLNF(det_parameters.model_location)

if not face_model.loaded_successfully:
    print("Error: Could not load the landmark detector")
    cap.release()
    sys.exit(-1)

if not face_model.eye_model:
    print("Warning: no eye model found")

while True:
    ret, rgb_image = cap.read()
    if not ret:
        break
    
    grayscale_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)
    grayscale_image = np.ubyte(grayscale_image)

    detection_success = of.DetectLandmarksInVideo(rgb_image, face_model,
                                               det_parameters, grayscale_image)

    landmarks = face_model.detected_landmarks
    #Reshape landmarks as a n by 2 matrix
    lm_np = np.array(landmarks).reshape(2, int(len(landmarks)/2)).T

    #Draw small circles on each landmark
    _ = np.apply_along_axis(lambda x : cv.circle(rgb_image, tuple(x), 3, (0, 0,
                                                                      255), -1), axis=1, arr=lm_np);

    cv.imshow('Landmarks',rgb_image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
