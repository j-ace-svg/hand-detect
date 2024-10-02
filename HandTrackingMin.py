import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.75,
                      max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

def calc_landmark_distance(fingerCoordinates, landmarkStart, landmarkEnd):
    return math.sqrt(
                      math.pow(fingerCoordinates[landmarkStart][0] - fingerCoordinates[landmarkEnd][0], 2) +
                      math.pow(fingerCoordinates[landmarkStart][1] - fingerCoordinates[landmarkEnd][1], 2) +
                      math.pow(fingerCoordinates[landmarkStart][2] - fingerCoordinates[landmarkEnd][2], 2)
                    )

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            fingerCoordinates = {}
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

            for id, landmark in enumerate(handLandmarks.landmark):
                #print(id, lm)
                height, width, c = img.shape
                centerx, centery, centerz = int(landmark.x*width), int(landmark.y*height), int(landmark.z*height)
                match id:
                    case 2: # Thumb knuckle
                        fingerCoordinates["thumb_k"] = (centerx, centery, centerz)
                    case 5: # Index tip
                        fingerCoordinates["index_k"] = (centerx, centery, centerz)
                    case 9: # Middle knuckle
                        fingerCoordinates["middle_k"] = (centerx, centery, centerz)
                    case 13: # Ring knuckle
                        fingerCoordinates["ring_k"] = (centerx, centery, centerz)
                    case 17: # Pinky knuckle
                        fingerCoordinates["pinky_k"] = (centerx, centery, centerz)
                    case 4: # Thumb tip
                        fingerCoordinates["thumb_t"] = (centerx, centery, centerz)
                    case 8: # Index tip
                        fingerCoordinates["index_t"] = (centerx, centery, centerz)
                    case 12: # Middle tip
                        fingerCoordinates["middle_t"] = (centerx, centery, centerz)
                    case 16: # Ring tip
                        fingerCoordinates["ring_t"] = (centerx, centery, centerz)
                    case 20: # Pinky tip
                        fingerCoordinates["pinky_t"] = (centerx, centery, centerz)
                    case _:
                        pass
                if id == 4:
                    cv2.circle(img, (centerx, centery), int(c / 1), (255, 0, 255), cv2.FILLED)
                cv2.putText(img,str(id), (centerx + 10, centery + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            
            # Pinch Gesture
            if fingerCoordinates.keys() >= {"thumb_t", "index_t"}:
                if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < 30):
                    print("pinch")
            
            # Vulcan Gesture
            if fingerCoordinates.keys() >= {"thumb_k", "index_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
                if (calc_landmark_distance(fingerCoordinates, "thumb_k", "index_k") > 70 and
                    calc_landmark_distance(fingerCoordinates, "index_t", "middle_t") < 50 and
                    calc_landmark_distance(fingerCoordinates, "middle_t", "ring_t") > 50 and
                    calc_landmark_distance(fingerCoordinates, "ring_t", "pinky_t") < 50):
                    print("vulcan")

            # OK Gesture
            if fingerCoordinates.keys() >= {"thumb_k", "index_k", "middle_k", "ring_k", "pinky_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
                if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < 30 and
                    calc_landmark_distance(fingerCoordinates, "middle_k", "middle_t") < 100 # continue conditions here
                    ):
                    pass

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img,str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == 27: # close window with 'ESC' key
        break