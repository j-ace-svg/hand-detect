import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time
import math
from operator import add, sub, mul
from functools import partial

EPSILON = 0.0001

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      model_complexity=1,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.75,
                      max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

def calc_landmark_distance(finger_coordinates, landmark_start, landmark_end):
    return math.sqrt(
                      pow(finger_coordinates[landmark_start][0] - finger_coordinates[landmark_end][0], 2) +
                      pow(finger_coordinates[landmark_start][1] - finger_coordinates[landmark_end][1], 2) +
                      pow(finger_coordinates[landmark_start][2] - finger_coordinates[landmark_end][2], 2)
                    ) / finger_coordinates["scalar"]

def process_hands(results):
    finger_coordinates = {}
    mp_draw.draw_landmarks(img, handLandmarks, mp_hands.HAND_CONNECTIONS)

    for id, landmark in enumerate(handLandmarks.landmark):
        #print(id, lm)
        height, width, c = img.shape
        centerx, centery, centerz = int(landmark.x*width), int(landmark.y*height), int(landmark.z*height)
        match id:
            case 0: # Wrist
                finger_coordinates["wrist"] = (centerx, centery, centerz)
            case 2: # Thumb knuckle
                finger_coordinates["thumb_k"] = (centerx, centery, centerz)
            case 5: # Index tip
                finger_coordinates["index_k"] = (centerx, centery, centerz)
            case 9: # Middle knuckle
                finger_coordinates["middle_k"] = (centerx, centery, centerz)
            case 13: # Ring knuckle
                finger_coordinates["ring_k"] = (centerx, centery, centerz)
            case 17: # Pinky knuckle
                finger_coordinates["pinky_k"] = (centerx, centery, centerz)
            case 4: # Thumb tip
                finger_coordinates["thumb_t"] = (centerx, centery, centerz)
            case 8: # Index tip
                finger_coordinates["index_t"] = (centerx, centery, centerz)
            case 12: # Middle tip
                finger_coordinates["middle_t"] = (centerx, centery, centerz)
            case 16: # Ring tip
                finger_coordinates["ring_t"] = (centerx, centery, centerz)
            case 20: # Pinky tip
                finger_coordinates["pinky_t"] = (centerx, centery, centerz)
            case _:
                pass
        if id in (5, 17, 0):
            cv2.circle(img, (centerx, centery), int(c / 1), (255, 0, 255), cv2.FILLED)
        cv2.putText(img,str(id), (centerx + 10, centery + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    
    # Cache distance scaling factor
    finger_coordinates["scalar"] = math.sqrt(
        pow(finger_coordinates["index_k"][0] - finger_coordinates["wrist"][0], 2) + 
        pow(finger_coordinates["index_k"][1] - finger_coordinates["wrist"][1], 2) + 
        pow(finger_coordinates["index_k"][2] - finger_coordinates["wrist"][2], 2)
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
    if finger_coordinates.keys() >= {"thumb_t", "index_t"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_t", "index_t") < gesture_distances["pinch-index"]):
            #print("pinch index")
            pinch_list[0] = True
            pass

    # Pinch middle Gesture
    if finger_coordinates.keys() >= {"thumb_t", "middle_t"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_t", "middle_t") < gesture_distances["pinch-middle"]):
            #print("pinch middle")
            pinch_list[1] = True

    # Pinch ring Gesture
    if finger_coordinates.keys() >= {"thumb_t", "ring_t"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_t", "ring_t") < gesture_distances["pinch-ring"]):
            #print("pinch ring")
            pinch_list[2] = True

    # Pinch pinky Gesture
    if finger_coordinates.keys() >= {"thumb_t", "pinky_t"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_t", "pinky_t") < gesture_distances["pinch-pinky"]):
            #print("pinch pinky")
            pinch_list[3] = True

    # Extend thumb Gesture
    if finger_coordinates.keys() >= {"thumb_t", "thumb_k"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_t", "index_k") > gesture_distances["split_thumb-index"] and
            calc_landmark_distance(finger_coordinates, "thumb_t", "wrist") > calc_landmark_distance(finger_coordinates, "thumb_k", "wrist")):
            #print("pinch pinky")
            extend_list[0] = True

    # Extend index Gesture
    if finger_coordinates.keys() >= {"index_t", "index_k"}:
        if (calc_landmark_distance(finger_coordinates, "index_t", "index_k") > gesture_distances["extend-index"] and
            calc_landmark_distance(finger_coordinates, "index_t", "wrist") > calc_landmark_distance(finger_coordinates, "index_k", "wrist")):
            #print("pinch pinky")
            extend_list[1] = True

    # Extend middle Gesture
    if finger_coordinates.keys() >= {"middle_t", "middle_k"}:
        if (calc_landmark_distance(finger_coordinates, "middle_t", "middle_k") > gesture_distances["extend-middle"] and
            calc_landmark_distance(finger_coordinates, "middle_t", "wrist") > calc_landmark_distance(finger_coordinates, "middle_k", "wrist")):
            #print("pinch pinky")
            extend_list[2] = True

    # Extend ring Gesture
    if finger_coordinates.keys() >= {"ring_t", "ring_k"}:
        if (calc_landmark_distance(finger_coordinates, "ring_t", "ring_k") > gesture_distances["extend-ring"] and
            calc_landmark_distance(finger_coordinates, "ring_t", "wrist") > calc_landmark_distance(finger_coordinates, "ring_k", "wrist")):
            #print("pinch pinky")
            extend_list[3] = True

    # Extend pinky Gesture
    if finger_coordinates.keys() >= {"pinky_t", "pinky_k"}:
        if (calc_landmark_distance(finger_coordinates, "pinky_t", "pinky_k") > gesture_distances["extend-pinky"] and
            calc_landmark_distance(finger_coordinates, "pinky_t", "wrist") > calc_landmark_distance(finger_coordinates, "pinky_k", "wrist")):
            #print("pinch pinky")
            extend_list[4] = True
    
    # Vulcan Gesture
    if finger_coordinates.keys() >= {"thumb_k", "index_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_k", "index_k") > gesture_distances["split_thumb-index"] and
            calc_landmark_distance(finger_coordinates, "index_t", "middle_t") < gesture_distances["clamp_index-middle"] and
            calc_landmark_distance(finger_coordinates, "middle_t", "ring_t") > gesture_distances["clamp_index-middle"] and
            calc_landmark_distance(finger_coordinates, "ring_t", "pinky_t") < gesture_distances["clamp_ring-pinky"]):
            #print("vulcan")
            pass

    # OK Gesture
    if finger_coordinates.keys() >= {"thumb_k", "index_k", "middle_k", "ring_k", "pinky_k", "thumb_t", "index_t", "middle_t", "ring_t", "pinky_t"}:
        if (calc_landmark_distance(finger_coordinates, "thumb_t", "index_t") < gesture_distances["pinch-index"] and
            calc_landmark_distance(finger_coordinates, "index_k", "middle_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(finger_coordinates, "index_k", "index_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(finger_coordinates, "middle_k", "middle_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(finger_coordinates, "ring_k", "ring_t") > gesture_distances["extend-index"] and
            calc_landmark_distance(finger_coordinates, "pinky_k", "pinky_t") > gesture_distances["extend-index"]):
            #print("ok")
            pass
        #print(calc_landmark_distance(finger_coordinates, "thumb_t", "index_k"))
        #print(pinch_list[0], pinch_list[1], pinch_list[2], pinch_list[3])
        #print(extend_list[0], extend_list[1], extend_list[2], extend_list[3], extend_list[4])
        #print(calc_landmark_distance(finger_coordinates, "thumb_t", "index_t") < 30,
        #    calc_landmark_distance(finger_coordinates, "index_k", "middle_t") > 100,
        #    calc_landmark_distance(finger_coordinates, "index_k", "index_t") > 100,
        #    calc_landmark_distance(finger_coordinates, "middle_k", "middle_t") > 100,
        #    calc_landmark_distance(finger_coordinates, "ring_k", "ring_t") > 100,
        #    calc_landmark_distance(finger_coordinates, "pinky_k", "pinky_t") > 100)

    hand_plane = []
    if finger_coordinates.keys() >= {"wrist"}:
        hand_plane.append(finger_coordinates["wrist"])
    else:
        hand_plane.append((0, 0, 0))
    if finger_coordinates.keys() >= {"index_k"}:
        hand_plane.append(finger_coordinates["index_k"])
    else:
        hand_plane.append((0, 0, 0))
    if finger_coordinates.keys() >= {"pinky_k"}:
        hand_plane.append(finger_coordinates["pinky_k"])
    else:
        hand_plane.append((0, 0, 0))

    return hand_plane, int(extend_list[0]) - int(extend_list[1])

class Ball():
    def __init__(self, startx, starty):
        self.x = startx
        self.y = starty
        self.velocity = [3, 3]
        self.last_contact_player = -1
        self.screen_width, self.screen_height = 0, 0
        self.radius = 5
        self.last_paddles = None
        self.drag_threshold = 12
        self.drag_proportion = 0.95
    
    def draw(self, img: cv2.typing.MatLike):
        cv2.circle(img, (int(self.x), int(self.y)), self.radius, (255, 0, 0), cv2.FILLED)

    def intersects_paddle(self, paddle_start, paddle_end, old_paddle_start, old_paddle_end):
        # Based on code from https://stackoverflow.com/a/67117213 (not anymore)
        if paddle_start == paddle_end:
            return False

        c1 = (self.x - self.velocity[0] * 2, self.y - self.velocity[1] * 2)
        c2 = (c1[0] + self.velocity[0] * 5, c1[1] + self.velocity[1] * 5)
        
        if old_paddle_start == old_paddle_end:
            def orientation(p, q, r):
                # Compute the orientation of the triplet (p, q, r)
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0  # Collinear
                return 1 if val > 0 else 2  # Clockwise or counterclockwise
            
            def do_intersect(p1, q1, p2, q2):
                # Find the four orientations needed for the general and special cases
                o1 = orientation(p1, q1, p2)
                o2 = orientation(p1, q1, q2)
                o3 = orientation(p2, q2, p1)
                o4 = orientation(p2, q2, q1)
                
                # General case
                if o1 != o2 and o3 != o4:
                    return True

                # Special cases (checking for collinear points)
                return False

            return do_intersect(c1, c2, paddle_start, paddle_end)
        # Original method (ball intersects paddle right now)
        '''
        x_linear = x2 - x1
        x_constant = x1 - cx
        y_linear = y2 - y1
        y_constant = y1 - cy
        a = x_linear * x_linear + y_linear * y_linear
        half_b = x_linear * x_constant + y_linear * y_constant
        c = x_constant * x_constant + y_constant * y_constant - r * r
        intersects = (
          half_b * half_b >= a * c and
          (-half_b <= a or c + half_b + half_b + a <= 0) and
          (half_b <= 0 or c <= 0)
        )
        '''
        # Second method (path of ball center over next frame intersects paddle)
        '''
        def ccw(A,B,C): # Check whether a group of points is sorted counterclockwise
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        
        intersects = ccw(paddle_start, c1, c2) != ccw(paddle_end, c1, c2) and ccw(paddle_start, paddle_end, c1) != ccw(paddle_start, paddle_end, c2)
        '''
        def is_point_in_quad(px, py, quad):
            # quad = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] represents the quadrilateral vertices
            
            def do_intersect(p1, p2, q1, q2):
                # Checks if the ray crosses a line segment (p1, p2)
                # Consider the line formed by q1q2
                if q1[1] > py and q2[1] > py or q1[1] < py and q2[1] < py:
                    return False
                if q1[0] < px and q2[0] < px:
                    return False
                if q1[1] > q2[1]:
                    q1, q2 = q2, q1  # Swap for upward direction

                intersect_x = q1[0] + (py - q1[1]) * (q2[0] - q1[0]) / (q2[1] - q1[1])
                return intersect_x > px

            # Check if the ray intersects with each edge of the quadrilateral
            intersections = 0
            for i in range(len(quad)):
                q1 = quad[i]
                q2 = quad[(i + 1) % len(quad)]
                if do_intersect(px, py, q1, q2):
                    intersections += 1
            
            # If the number of intersections is odd, the point is inside
            return intersections % 2 == 1

        # Return if either point is inside the quadrilateral path
        c1_inside = is_point_in_quad(*c1, [paddle_start, paddle_end, old_paddle_start, old_paddle_end])
        c2_inside = is_point_in_quad(*c1, [paddle_start, paddle_end, old_paddle_start, old_paddle_end])
        if c1_inside or c2_inside:
            return True
        
        def orientation(p, q, r):
            # Compute the orientation of the triplet (p, q, r)
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or counterclockwise

        def do_intersect(p1, q1, p2, q2):
            # Find the four orientations needed for the general and special cases
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
            
            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special cases (checking for collinear points)
            return False

        # Check if the line segment intersects any of the four edges
        path_intersects_quad = (
            do_intersect(c1, c2, paddle_start, paddle_end) or
            do_intersect(c1, c2, paddle_end, old_paddle_end) or
            do_intersect(c1, c2, old_paddle_end, old_paddle_start) or
            do_intersect(c1, c2, old_paddle_start, paddle_start)
        )
        
        # Return if the paddle intersects any of the sides of the quadrilateral path

        return path_intersects_quad

    def paddle_bounce(self, paddles):
        if self.last_paddles == None:
            self.last_paddles = paddles
        for player, (paddle, old_paddle) in enumerate(zip(paddles, self.last_paddles)):
            if self.last_contact_player == player or not self.intersects_paddle(*paddle, *old_paddle):
                continue
            paddle_vec = tuple(map(sub, paddle[0], paddle[1]))
            paddle_vec_mag = math.sqrt(sum(map(partial(pow, exp=2), paddle_vec)))
            if abs(paddle_vec_mag) < EPSILON:
                paddle_vec_mag = EPSILON * 1 if paddle_vec_mag > 0 else -1
            paddle_vec_normalized = tuple(map(partial(mul, 1/paddle_vec_mag), paddle_vec))

            paddle_norm_left = (-paddle_vec_normalized[1], paddle_vec_normalized[0])
            dot_prod = sum(map(mul, self.velocity, paddle_norm_left))

            velocity_start = tuple(map(sub, paddle[0], old_paddle[0]))
            velocity_end = tuple(map(sub, paddle[1], old_paddle[1]))
            velocity_avg = tuple(map(partial(mul, 1/2), map(add, velocity_start, velocity_end)))
            velocity_normal_force = tuple(map(partial(mul, sum(map(mul, velocity_avg, paddle_norm_left))), paddle_norm_left))

            deflection = tuple(map(partial(mul, (-2 * dot_prod)), paddle_norm_left))
            print(player)
            new_velocity = list(map(add, map(add, self.velocity, deflection), velocity_normal_force))
            if player == 0: # Only register the paddle hit if in the right direction
                if new_velocity[0] > self.velocity[0]:
                    self.velocity = new_velocity
                    self.last_contact_player = player
            else:
                if new_velocity[0] < self.velocity[0]:
                    self.velocity = new_velocity
                    self.last_contact_player = player
        self.last_paddles = paddles
        self.cool_off_speed()
    
    def cool_off_speed(self):
        speed_squared = sum(map(partial(pow, exp=2), self.velocity))
        if speed_squared > self.drag_threshold ** 2:
            print("drag")
            old_speed = math.sqrt(speed_squared)
            speed_drag = old_speed - self.drag_threshold
            new_speed = self.drag_threshold + speed_drag * self.drag_proportion
            new_velocity = list(map(partial(mul, new_speed / old_speed), self.velocity))
            self.velocity = new_velocity
    
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

    #paddle_direction = (point_a[0] - hand_center[0], point_a[1] - hand_center[1])
    paddle_direction = tuple(map(sub, point_a, hand_center))
    paddle_direction_magnitude = math.sqrt(sum(map(partial(pow, exp=2), paddle_direction)))
    paddle_direction_normalized = tuple(map(partial(mul, 1/max(paddle_direction_magnitude, EPSILON)), paddle_direction))

    paddle_size = 40

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
    
    #print(movement, plane_coords)
    paddle_coords = [calc_paddle(hand) for hand in plane_coords]
    #print(paddle_coords)
    assert(len(paddle_coords[0][0]) == 2)

    ball.update()
    ball.paddle_bounce(paddle_coords)
    ball.draw(img)
    for paddle in paddle_coords:
        paddle_int = [tuple(map(int, point)) for point in paddle]
        cv2.line(img, paddle_int[0], paddle_int[1], (0, 0, 255), 2)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == 27: # close window with 'ESC' key
        break
