import cv2
import numpy as np

# Importing the image Query.
image_query = cv2.imread("data/viewQuery.JPG")

# Initializing the video to draw trackers
cap = cv2.VideoCapture("data/view.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data/out.mp4', fourcc, 20.0, (2337, 1012))
# Changing the Query image into RGB
image_queryRGB = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
# Changing the Query image into Grayscale for training
image_queryGray = cv2.cvtColor(image_query, cv2.COLOR_RGB2GRAY)
# Initializing Oriented Fast and Rotational Brief
orb = cv2.ORB_create(500, 2.1)
# Detecting keypoints in or Query image
query_trackers, query_descriptor = orb.detectAndCompute(image_queryGray, None)
# Copying the Query image to draw keypoints on them
queryTrackImage = np.copy(image_queryRGB)
# Setting out threshold Value.
threshold = 11
# Running throught the Video

while True:
    # Reading in the Video
    ret, frame = cap.read()
    if ret:
        # Changing the frame color to gray for training purposes
        train_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Detecting the keypoints in the in the video frames.
        keypoints_train, descriptor_train = orb.detectAndCompute(train_gray,
                                                                 None)
        # Initializing the crossChecking algorithm
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Finding the matching vectors
        matches = bf.match(descriptor_train, query_descriptor)
        matches = sorted(matches, key=lambda x: x.distance)
        # Drawing the matching vectors.
        result = cv2.drawMatches(frame,
                                 keypoints_train,
                                 image_queryRGB,
                                 query_trackers,
                                 matches[:threshold],
                                 image_queryGray,
                                 flags=2)
        cv2.imshow('frame', result)
        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
