import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time
import math
from operator import add, sub, mul
from functools import partial

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.75,
                      max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

def calc_landmark_distance(fingerCoordinates, landmarkStart, landmarkEnd):
    return math.sqrt(
                      math.pow(fingerCoordinates[landmarkStart][0] - fingerCoordinates[landmarkEnd][0], 2) +
                      math.pow(fingerCoordinates[landmarkStart][1] - fingerCoordinates[landmarkEnd][1], 2) +
                      math.pow(fingerCoordinates[landmarkStart][2] - fingerCoordinates[landmarkEnd][2], 2)
                    ) / fingerCoordinates["scalar"]

def process_hands(results):
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
        if id in (5, 17, 0):
            cv2.circle(img, (centerx, centery), int(c / 1), (255, 0, 255), cv2.FILLED)
        cv2.putText(img,str(id), (centerx + 10, centery + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    
    # Cache distance scaling factor
    fingerCoordinates["scalar"] = math.sqrt(
        math.pow(fingerCoordinates["index_k"][0] - fingerCoordinates["wrist"][0], 2) + 
        math.pow(fingerCoordinates["index_k"][1] - fingerCoordinates["wrist"][1], 2) + 
        math.pow(fingerCoordinates["index_k"][2] - fingerCoordinates["wrist"][2], 2)
    )

    gesture_distances = {
        "pinch-index": 0.24,
        "pinch-middle": 0.26,
        "pinch-ring": 0.24,
        "pinch-pinky": 0.26,
        "extend-thumb": 0.45,
        "extend-index": 0.45,
        "extend-middle": 0.43,
        "extend-ring": 0.45,
        "extend-pinky": 0.45,
        "clamp_index-middle": 0.28,
        "clamp_ring-pinky": 0.33,
        "split_thumb-index": 0.50
    }

    pinch_list = [False, False, False, False]
    extend_list = [False, False, False, False, False]

    # Pinch index Gesture
    if fingerCoordinates.keys() >= {"thumb_t", "index_t"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < gesture_distances["pinch-index"]):
            #print("pinch index")
            pinch_list[0] = True
            pass

    # Pinch middle Gesture
    if fingerCoordinates.keys() >= {"thumb_t", "middle_t"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_t", "middle_t") < gesture_distances["pinch-middle"]):
            #print("pinch middle")
            pinch_list[1] = True

    # Pinch ring Gesture
    if fingerCoordinates.keys() >= {"thumb_t", "ring_t"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_t", "ring_t") < gesture_distances["pinch-ring"]):
            #print("pinch ring")
            pinch_list[2] = True

    # Pinch pinky Gesture
    if fingerCoordinates.keys() >= {"thumb_t", "pinky_t"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_t", "pinky_t") < gesture_distances["pinch-pinky"]):
            #print("pinch pinky")
            pinch_list[3] = True

    # Extend thumb Gesture
    if fingerCoordinates.keys() >= {"thumb_t", "thumb_k"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_k") > gesture_distances["split_thumb-index"] and
            calc_landmark_distance(fingerCoordinates, "thumb_t", "wrist") > calc_landmark_distance(fingerCoordinates, "thumb_k", "wrist")):
            #print("pinch pinky")
            extend_list[0] = True

    # Extend index Gesture
    if fingerCoordinates.keys() >= {"index_t", "index_k"}:
        if (calc_landmark_distance(fingerCoordinates, "index_t", "index_k") > gesture_distances["extend-index"] and
            calc_landmark_distance(fingerCoordinates, "index_t", "wrist") > calc_landmark_distance(fingerCoordinates, "index_k", "wrist")):
            #print("pinch pinky")
            extend_list[1] = True

    # Extend middle Gesture
    if fingerCoordinates.keys() >= {"middle_t", "middle_k"}:
        if (calc_landmark_distance(fingerCoordinates, "middle_t", "middle_k") > gesture_distances["extend-middle"] and
            calc_landmark_distance(fingerCoordinates, "middle_t", "wrist") > calc_landmark_distance(fingerCoordinates, "middle_k", "wrist")):
            #print("pinch pinky")
            extend_list[2] = True

    # Extend ring Gesture
    if fingerCoordinates.keys() >= {"ring_t", "ring_k"}:
        if (calc_landmark_distance(fingerCoordinates, "ring_t", "ring_k") > gesture_distances["extend-ring"] and
            calc_landmark_distance(fingerCoordinates, "ring_t", "wrist") > calc_landmark_distance(fingerCoordinates, "ring_k", "wrist")):
            #print("pinch pinky")
            extend_list[3] = True

    # Extend pinky Gesture
    if fingerCoordinates.keys() >= {"pinky_t", "pinky_k"}:
        if (calc_landmark_distance(fingerCoordinates, "pinky_t", "pinky_k") > gesture_distances["extend-pinky"] and
            calc_landmark_distance(fingerCoordinates, "pinky_t", "wrist") > calc_landmark_distance(fingerCoordinates, "pinky_k", "wrist")):
            #print("pinch pinky")
            extend_list[4] = True
    
    # Vulcan Gesture
    if fingerCoordinates.keys() >= {"thumb_k", "index_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_k", "index_k") > gesture_distances["split_thumb-index"] and
            calc_landmark_distance(fingerCoordinates, "index_t", "middle_t") < gesture_distances["clamp_index-middle"] and
            calc_landmark_distance(fingerCoordinates, "middle_t", "ring_t") > gesture_distances["clamp_index-middle"] and
            calc_landmark_distance(fingerCoordinates, "ring_t", "pinky_t") < gesture_distances["clamp_ring-pinky"]):
            #print("vulcan")
            pass

    # OK Gesture
    if fingerCoordinates.keys() >= {"thumb_k", "index_k", "middle_k", "ring_k", "pinky_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
        if (calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < gesture_distances["pinch-index"] and
            calc_landmark_distance(fingerCoordinates, "index_k", "middle_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(fingerCoordinates, "index_k", "index_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(fingerCoordinates, "middle_k", "middle_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(fingerCoordinates, "ring_k", "ring_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(fingerCoordinates, "pinky_k", "pinky_t") > gesture_distances["extend-index"]):
            #print("ok")
            pass
        #print(calc_landmark_distance(fingerCoordinates, "thumb_t", "index_k"))
        #print(pinch_list[0], pinch_list[1], pinch_list[2], pinch_list[3])
        #print(extend_list[0], extend_list[1], extend_list[2], extend_list[3], extend_list[4])
        #print(calc_landmark_distance(fingerCoordinates, "thumb_t", "index_t") < 30,
        #    calc_landmark_distance(fingerCoordinates, "index_k", "middle_t") > 100,
        #    calc_landmark_distance(fingerCoordinates, "index_k", "index_t") > 100,
        #    calc_landmark_distance(fingerCoordinates, "middle_k", "middle_t") > 100,
        #    calc_landmark_distance(fingerCoordinates, "ring_k", "ring_t") > 100,
        #    calc_landmark_distance(fingerCoordinates, "pinky_k", "pinky_t") > 100)

    hand_plane = []
    if fingerCoordinates.keys() >= {"wrist"}:
        hand_plane.append(fingerCoordinates["wrist"])
    else:
        hand_plane.append((0, 0, 0))
    if fingerCoordinates.keys() >= {"index_k"}:
        hand_plane.append(fingerCoordinates["index_k"])
    else:
        hand_plane.append((0, 0, 0))
    if fingerCoordinates.keys() >= {"pinky_k"}:
        hand_plane.append(fingerCoordinates["pinky_k"])
    else:
        hand_plane.append((0, 0, 0))

    return hand_plane, int(extend_list[0]) - int(extend_list[1])

class Ball():
    def __init__(self, startx, starty):
        self.x = startx
        self.y = starty
        self.velocity = [3, 3]
        self.screen_width, self.screen_height = 0, 0
        self.radius = 5
    
    def draw(self, img: cv2.typing.MatLike):
        cv2.circle(img, (self.x, self.y), self.radius, (255, 0, 0), cv2.FILLED)
    
    def update(self):
        try:
            _, _, self.screen_width, self.screen_height = cv2.getWindowImageRect("Image")
        except:
            pass
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        if self.x + self.radius + self.velocity[0] > self.screen_width or self.x - self.radius + self.velocity[0] < 0:
            self.velocity[0] = -self.velocity[0]
        if self.y + self.radius + self.velocity[1] > self.screen_height or self.y - self.radius + self.velocity[1] < 0:
            self.velocity[1] = -self.velocity[1]

def calc_paddle(plane_coords):
    point_a = plane_coords[0]
    point_b = plane_coords[1]
    point_c = plane_coords[2]

    #vector_a_b = (point_b[0] - point_a[0], point_b[1] - point_a[1], point_b[2] - point_a[2])
    #vector_a_c = (point_c[0] - point_a[0], point_c[1] - point_a[1], point_c[2] - point_a[2])

    #cross_product = ([vector_a_b[1]*vector_a_c[2] - vector_a_b[2]*vector_a_c[1],
    #                  vector_a_b[2]*vector_a_c[0] - vector_a_b[0]*vector_a_c[2],
    #                  vector_a_b[0]*vector_a_c[1] - vector_a_b[1]*vector_a_c[0]])

    #coef_x, coef_y, coef_z = cross_product
    #plane_const = (coef_x * point_a[0]) + (coef_y * point_a[1]) + (coef_z * point_a[2])

    hand_center = ((2 * point_a[0] + point_b[0] + point_c[0]) / 4,
                  (2 * point_a[1] + point_b[1] + point_c[1]) / 4)

    paddle_direction = (point_a[0] - hand_center[0], point_a[1] - hand_center[1])
    paddle_direction = map(sub, point_a, hand_center)
    paddle_direction_magnitude = sqrt(sum(map(partial(math.pow, y=2), paddle_direction)))
    paddle_direction_normalized = map(partial(mul, 1/paddle_direction_magnitude), paddle_direction)

    paddle_size = 1

    paddle_start = tuple(map(sub, hand_center, map(partial(mul, paddle_size), paddle_direction_normalized)))
    paddle_end = tuple(map(add, hand_center, map(partial(mul, paddle_size), paddle_direction_normalized)))

    return [paddle_start, paddle_end]

ball = Ball(20, 20)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    movement = [0, 0]

    plane_coords = [[(0, 0, 0), (0, 0, 0), (0, 0, 0)], [(0, 0, 0), (0, 0, 0), (0, 0, 0)]]

    if results.multi_hand_landmarks:
        for handLandmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness_dict = MessageToDict(handedness)
            plane_coords[handedness_dict["classification"][0]["index"]], movement[handedness_dict["classification"][0]["index"]] = process_hands(results)
    
    print(movement, plane_coords)
    paddle_coords = [calc_paddle(hand) for hand in plane_coords]

    ball.update()
    ball.draw(img)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == 27: # close window with 'ESC' key
        break
