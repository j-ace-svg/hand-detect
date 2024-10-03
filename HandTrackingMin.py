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
                    ) / fingerCoordinates["scalar"]

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
                    case 0: # Wrist
                        fingerCoordinates["wrist"] = (centerx, centery, centerz)
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
            
            # Cache distance scaling factor
            fingerCoordinates["scalar"] = math.sqrt(
                math.pow(fingerCoordinates["index_k"][0] - fingerCoordinates["wrist"][0], 2) + 
                math.pow(fingerCoordinates["index_k"][1] - fingerCoordinates["wrist"][1], 2) + 
                math.pow(fingerCoordinates["index_k"][2] - fingerCoordinates["wrist"][2], 2)
            )

            gesture_distances = {
                "pinch": 0.21,
                "extend": 0.45,
                "clamp_index-middle": 0.28,
                "clamp_ring-pinky": 0.33,
                "split_thumb-index": 0.315
            }

            # Pinch Gesture
            if fingerCoordinates.keys() >= {"thumb_t", "index_t"}:
                if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < gesture_distances["pinch"]):
                    print("pinch")
                    pass
            
            # Vulcan Gesture
            if fingerCoordinates.keys() >= {"thumb_k", "index_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
                if (calc_landmark_distance(fingerCoordinates, "thumb_k", "index_k") > gesture_distances["split_thumb-index"] and
                    calc_landmark_distance(fingerCoordinates, "index_t", "middle_t") < gesture_distances["clamp_index-middle"] and
                    calc_landmark_distance(fingerCoordinates, "middle_t", "ring_t") > gesture_distances["clamp_index-middle"] and
                    calc_landmark_distance(fingerCoordinates, "ring_t", "pinky_t") < gesture_distances["clamp_ring-pinky"]):
                    print("vulcan")

            # OK Gesture
            if fingerCoordinates.keys() >= {"thumb_k", "index_k", "middle_k", "ring_k", "pinky_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
                if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < gesture_distances["pinch"] and
                    calc_landmark_distance(fingerCoordinates, "index_k", "middle_t") > gesture_distances["extend"] and
                    calc_landmark_distance(fingerCoordinates, "index_k", "index_t") > gesture_distances["extend"] and
                    calc_landmark_distance(fingerCoordinates, "middle_k", "middle_t") > gesture_distances["extend"] and
                    calc_landmark_distance(fingerCoordinates, "ring_k", "ring_t") > gesture_distances["extend"] and
                    calc_landmark_distance(fingerCoordinates, "pinky_k", "pinky_t") > gesture_distances["extend"]):
                    print("ok")
                    #pass
                print(calc_landmark_distance(fingerCoordinates, "ring_t", "pinky_t"))
                #print(calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < 30,
                #    calc_landmark_distance(fingerCoordinates, "index_k", "middle_t") > 100,
                #    calc_landmark_distance(fingerCoordinates, "index_k", "index_t") > 100,
                #    calc_landmark_distance(fingerCoordinates, "middle_k", "middle_t") > 100,
                #    calc_landmark_distance(fingerCoordinates, "ring_k", "ring_t") > 100,
                #    calc_landmark_distance(fingerCoordinates, "pinky_k", "pinky_t") > 100)

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img,str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == 27: # close window with 'ESC' key
        break