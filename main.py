import threading
import time
import mediapipe as mp
import cv2
import numpy as np
from deepface import DeepFace
import playsound
import threading


IMAGE_SKULL = 'skull1.png'


def check_emotion(image):

    global IMAGE_SKULL

    try:
        face_analysis = DeepFace.analyze(image, enforce_detection=False)[0]
        # print(face_analysis)
        if face_analysis['dominant_emotion'] == 'happy' or face_analysis['dominant_emotion'] == 'surprise' or \
                face_analysis['emotion']['happy'] > 5:
            IMAGE_SKULL = 'skull3.jpg'
            t1 = threading.Thread(target=play_laugh)
            t1.start()
        else:
            IMAGE_SKULL = 'skull1.png'
    except Exception as e:
        # print("HI")
        print(e)


def play_laugh():
    # playsound.playsound('laugh.mp3')
    return

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

print("INITIALISING...")
face_analysis = DeepFace.analyze('skull1.png', enforce_detection=False)[0]
print("DONE")

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks

cap = cv2.VideoCapture(0)

prev_time = time.time()

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Creating a blank image
        b_img = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
        skull_img = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
        skull_img.fill(255)

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if time.time() - prev_time >= 5:
            prev_time = time.time()
            # try:
            #     face_analysis = DeepFace.analyze(image, enforce_detection=False)[0]
            #     # print(face_analysis)
            #     if face_analysis['dominant_emotion'] == 'happy' or face_analysis['dominant_emotion'] == 'surprise' or face_analysis['emotion']['happy'] > 5:
            #         IMAGE_SKULL = 'skull3.jpg'
            #         t1 = threading.Thread(target=play_laugh)
            #         t1.start()
            #     else:
            #         IMAGE_SKULL = 'skull1.png'
            # except Exception as e:
            #     # print("HI")
            #     print(e)
            #     break
            checker_thread = threading.Thread(target=check_emotion, args=(image,))
            checker_thread.start()

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(b_img, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        centroid = [0, 0]

        x_max = y_max = -1
        x_min = y_min = 501

        if results.face_landmarks is not None:

            cnt = 0

            for landmark in results.face_landmarks.landmark:

                cnt += 1
                x = landmark.x
                y = landmark.y

                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])

                centroid[0] += relative_x
                centroid[1] += relative_y

                x_max = max(x_max, relative_x)
                x_min = min(x_min, relative_x)
                y_max = max(y_max, relative_y)
                y_min = min(y_min, relative_y)

            centroid[0] /= cnt
            centroid[1] /= cnt

            cv2.circle(image, (int(centroid[0]), int(centroid[1])), radius=10, color=(225, 0, 100), thickness=1)

            x_min -= 20
            y_min -= 20
            x_max += 20
            y_max += 20

            x_offset, y_offset = (x_min, y_min)

            s_img = cv2.imread(IMAGE_SKULL)
            s_img = cv2.resize(s_img, (x_max-x_min, y_max-y_min))

            # image[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
            for i in range(y_offset, min(y_offset+s_img.shape[0], image.shape[0])):
                for j in range(x_offset, min(x_offset + s_img.shape[1], image.shape[1])):
                    skull_img[i][j] = s_img[i-y_offset][j-x_offset]
                    # image[i][j] = s_img[i-y_offset][j-x_offset]


        # CREATING CHEST

        # chest_x_max = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1])
        # chest_x_min = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1])
        chest_x_min = x_min - 30
        chest_x_max = x_max + 30
        # chest_y_upper = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])
        chest_y_upper = y_max + 10
        chest_y_lower = int(chest_y_upper + 1.5*(y_max - y_min))

        chest_top_left = (chest_x_min, chest_y_upper)
        chest_top_right = (chest_x_max, chest_y_upper)

        # print(chest_x_min, chest_x_max, chest_y_upper, chest_y_lower)
        IMAGE_CHEST = 'chest.jpg'
        s_img = cv2.imread(IMAGE_CHEST)

        try:
            s_img = cv2.resize(s_img, (chest_x_max - chest_x_min, chest_y_lower - chest_y_upper))
        except:
            pass

        for i in range(chest_y_upper, chest_y_lower):
            for j in range(chest_x_min, chest_x_max):
                try:
                    skull_img[i][j] = 0
                    skull_img[i][j] = s_img[i-chest_y_lower][j-chest_x_max]
                    # image[i][j] =
                except:
                    pass


        # break

        # 2. Right hand
        mp_drawing.draw_landmarks(b_img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        mp_drawing.draw_landmarks(skull_img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(219, 112, 147), thickness=1, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                                  )


        # RIGHT ELBOW
        try:
            right_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image.shape[1])
            right_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image.shape[0])
            right_wrist_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image.shape[1])
            right_wrist_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image.shape[0])

            # MARKING POINTS
            cv2.circle(image, (right_elbow_x, right_elbow_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (right_elbow_x, right_elbow_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (right_wrist_x, right_wrist_y), radius=10, color=(225, 0, 100), thickness=1)

            # Black color in BGR
            color = (0, 0, 0)

            # Line thickness of 9 px
            thickness = 5

            # DRAW HANDS
            cv2.line(skull_img, (chest_top_left[0] + 30, chest_top_left[1] + 20), (right_elbow_x, right_elbow_y), color, thickness)
            cv2.line(skull_img, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), color, thickness)

        except:
            pass


        # 3. Left Hand
        mp_drawing.draw_landmarks(b_img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        mp_drawing.draw_landmarks(skull_img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(219, 112, 147), thickness=1, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                                  )

        # LEFT ELBOW
        try:
            left_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1])
            left_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])
            left_wrist_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image.shape[1])
            left_wrist_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image.shape[0])

            # MARKING POINTS
            cv2.circle(image, (left_elbow_x, left_elbow_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (left_elbow_x, left_elbow_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (left_wrist_x, left_wrist_y), radius=10, color=(225, 0, 100), thickness=1)

            # Black color in BGR
            color = (0, 0, 0)

            # Line thickness of 9 px
            thickness = 5

            # DRAW HANDS
            cv2.line(skull_img, (chest_top_right[0] - 30, chest_top_right[1] + 20), (left_elbow_x, left_elbow_y), color, thickness)
            cv2.line(skull_img, (left_elbow_x, left_elbow_y), (left_wrist_x, left_wrist_y), color, thickness)

        except:
            pass

        # 4. Pose Detections
        mp_drawing.draw_landmarks(b_img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )


        # 5. LEFT LEG
        try:

            right_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1])
            right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])

            right_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.shape[1])
            right_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.shape[0])

            right_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image.shape[1])
            right_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image.shape[0])

            # MARKING POINTS
            cv2.circle(image, (right_hip_x, right_hip_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(image, (right_knee_x, right_knee_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(image, (right_ankle_x, right_ankle_y), radius=10, color=(225, 0, 100), thickness=1)

            cv2.circle(skull_img, (right_hip_x, right_hip_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (right_knee_x, right_knee_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (right_ankle_x, right_ankle_y), radius=10, color=(225, 0, 100), thickness=1)

            # Black color in BGR
            color = (0, 0, 0)

            # Line thickness of 9 px
            thickness = 5

            # DRAW LEGS
            cv2.line(skull_img, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), color, thickness)
            cv2.line(skull_img, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), color, thickness)

        except:
            pass


        # 6. RIGHT LEG
        try:

            left_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1])
            left_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])

            left_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1])
            left_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0])

            left_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image.shape[1])
            left_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image.shape[0])

            # MARKING POINTS
            cv2.circle(image, (left_hip_x, left_hip_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(image, (left_knee_x, left_knee_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(image, (left_ankle_x, left_ankle_y), radius=10, color=(225, 0, 100), thickness=1)

            cv2.circle(skull_img, (left_hip_x, left_hip_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (left_knee_x, left_knee_y), radius=10, color=(225, 0, 100), thickness=1)
            cv2.circle(skull_img, (left_ankle_x, left_ankle_y), radius=10, color=(225, 0, 100), thickness=1)

            # Black color in BGR
            color = (0, 0, 0)

            # Line thickness of 9 px
            thickness = 5

            # DRAW LEGS
            cv2.line(skull_img, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), color, thickness)
            cv2.line(skull_img, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), color, thickness)
        except:
            pass


        cv2.imshow('Raw Webcam Feed', image)
        cv2.imshow('Drawing', b_img)
        cv2.imshow('Skull Image', skull_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
