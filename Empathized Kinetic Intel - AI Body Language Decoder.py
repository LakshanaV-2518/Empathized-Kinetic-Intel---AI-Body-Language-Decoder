#!/usr/bin/env python
# coding: utf-8

# # Step-0 : Install and Import Dependencies

# In[2]:


get_ipython().system('pip install mediapipe opencv-python pandas scikit-learn')

get_ipython().system('pip install protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')


# In[4]:


import mediapipe as mp # Import mediapipe
import cv2 # Import opencv

print(mp.__version__)  # Print the installed Mediapipe version
print("Mediapipe is working correctly.")

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions


# # Step-1 : Get Realtime Web Cam Feed & Make Some Detections

# In[6]:


get_ipython().system('pip install protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[8]:


results


# In[10]:


results.face_landmarks


# In[12]:


results.right_hand_landmarks


# In[14]:


results.left_hand_landmarks


# In[28]:


results.pose_landmarks


# In[16]:


results.face_landmarks.landmark[0].visibility


# In[18]:


results.pose_landmarks.landmark[0].visibility


# # Step-2 : Capture Landmarks & Export to CSV
# 
# <img src="https://i.imgur.com/8bForKY.png">
# <img src="https://i.imgur.com/AzKNp7A.png">

# In[20]:


import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# In[22]:


num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
num_coords


# In[24]:


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# In[26]:


landmarks


# In[50]:


with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[52]:


class_name = "Happy"


# In[54]:


get_ipython().system('pip install protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

         # Export coordinates
        try:
            # Extract Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face Landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten())

            # Concat Rows
            row = pose_row+face_row

            # Append Class Name
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
        
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[56]:


class_name = "Sad"


# In[58]:


get_ipython().system('pip install protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

         # Export coordinates
        try:
            # Extract Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face Landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten())

            # Concat Rows
            row = pose_row+face_row

            # Append Class Name
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
        
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[66]:


class_name = "Victorious"


# In[68]:


get_ipython().system('pip install protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

         # Export coordinates
        try:
            # Extract Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face Landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten())

            # Concat Rows
            row = pose_row+face_row

            # Append Class Name
            row.insert(0, class_name)

            
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
        
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[80]:


class_name = "RigidPosture"


# In[82]:


import cv2
import numpy as np
import csv
import warnings
import mediapipe as mp

get_ipython().system('pip install protobuf==3.20.3')

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize webcam feed
cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define a threshold for what constitutes a "rigid" posture.
# For example, if the change in landmark positions is below this threshold over multiple frames, we consider the posture rigid.
rigid_threshold = 0.01  # Adjust as needed

# Variables to store the previous frame's pose landmarks
prev_pose_landmarks = None

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:
    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks (same as before)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            
            # If we have a previous frame to compare, calculate the differences
            if prev_pose_landmarks:
                rigid_posture = True  # Assume rigid posture by default

                # Check movement in key landmarks (e.g., shoulders and spine)
                for idx in [11, 12, 23, 24, 25, 26]:  # Landmarks for shoulders, hip and knees
                    prev_landmark = prev_pose_landmarks[idx]
                    curr_landmark = pose_landmarks[idx]

                    # Calculate the movement (Euclidean distance)
                    movement = np.sqrt((curr_landmark.x - prev_landmark.x) ** 2 + (curr_landmark.y - prev_landmark.y) ** 2)

                    # If any key landmark moves beyond the threshold, it's not a rigid posture
                    if movement > rigid_threshold:
                        rigid_posture = False
                        break

                # If rigid posture detected, mark it and save to CSV
                rigid_posture_label = 1 if rigid_posture else 0
                
                # Prepare row for CSV
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())
                row = [class_name, rigid_posture_label] + pose_row
                
                # Export to CSV
                with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            
            # Update previous landmarks for the next frame comparison
            prev_pose_landmarks = pose_landmarks
        
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[102]:


class_name = "SpiderFingers"


# In[104]:


import cv2
import csv
import numpy as np
import mediapipe as mp
import warnings

get_ipython().system('pip install protobuf==3.20.3')

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize MediaPipe holistic and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

# Function to detect spider fingers
def detect_spider_fingers(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]

    # Calculate distances between the fingertips
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)

    # Logic to detect spider fingers (adjust thresholds as needed)
    if thumb_index_dist > 0.03 and thumb_pinky_dist > 0.05:
        return True
    return False

# Capture video from webcam (or IP camera)
cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initialize the holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        # Export coordinates
        try:
            # Check for spider fingers on the right hand
            if results.right_hand_landmarks and detect_spider_fingers(results.right_hand_landmarks):
                # Extract Pose Landmarks
                pose = results.pose_landmarks.landmark if results.pose_landmarks else []
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()) if pose else []

                # Extract Face Landmarks
                face = results.face_landmarks.landmark if results.face_landmarks else []
                face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten()) if face else []

                # Concat Rows
                row = pose_row + face_row

                # Append Class Name
                row.insert(0, class_name)

                # Export to CSV
                with open('coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

        except Exception as e:
            print(f"Error occurred: {e}")
        
        # Display the video feed
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[128]:


class_name = "HandTrumpet"


# In[130]:


get_ipython().system('pip install protobuf==3.20.3')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Function to detect Hand Trumpet gesture
def is_hand_trumpet(landmarks):
    # Define the landmarks for the right hand
    thumb_tip = landmarks[4]   # Tip of thumb
    index_tip = landmarks[8]   # Tip of index finger
    middle_tip = landmarks[12]  # Tip of middle finger
    ring_tip = landmarks[16]    # Tip of ring finger
    pinky_tip = landmarks[20]   # Tip of pinky finger

    # Get x, y coordinates of the thumb and finger tips
    thumb_x, thumb_y = thumb_tip.x, thumb_tip.y
    index_x, index_y = index_tip.x, index_tip.y
    middle_x, middle_y = middle_tip.x, middle_tip.y
    ring_x, ring_y = ring_tip.x, ring_tip.y
    pinky_x, pinky_y = pinky_tip.x, pinky_tip.y

    # Define thresholds for proximity (this can be adjusted based on testing)
    proximity_threshold = 0.1  # Adjust based on scale (0.0 to 1.0)

    # Check if the fingers are close to each other
    is_proximity = (
        abs(index_x - middle_x) < proximity_threshold and
        abs(index_x - ring_x) < proximity_threshold and
        abs(index_x - pinky_x) < proximity_threshold
    )
    
    # Check if the thumb is positioned appropriately
    thumb_position = thumb_y < index_y  # Thumb should be above the index finger

    # Combine conditions to detect hand trumpet gesture
    if is_proximity and thumb_position:
        return True
    return False


cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        # Export coordinates
        try:
            # Extract Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face Landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten())

            # Concat Rows
            row = pose_row+face_row

            # Detect "Hand Trumpet" Gesture
            if is_hand_trumpet(results.right_hand_landmarks.landmark):
                print("Hand Trumpet Gesture Detected!")
                row.insert(0, "Hand Stims (Hand Trumpet)")
            else:
                row.insert(0, class_name)

            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
        
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[140]:


class_name = "BirdWings"


# In[142]:


import cv2
import mediapipe as mp
import numpy as np
import csv
import warnings

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define a function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Define logic for detecting bird wings behavior
def is_bird_wings(right_hand_landmarks, left_hand_landmarks, pose_landmarks):
    """Detect bird wings hand stimming behavior."""
    
    # Get shoulder, elbow, and wrist positions for both sides
    right_shoulder = pose_landmarks[12]
    right_elbow = pose_landmarks[14]
    right_wrist = right_hand_landmarks[0]  # Index 0 is the wrist for the right hand

    left_shoulder = pose_landmarks[11]
    left_elbow = pose_landmarks[13]
    left_wrist = left_hand_landmarks[0]  # Index 0 is the wrist for the left hand

    # Calculate distances between shoulder and wrist (to check outward hand movement)
    right_hand_to_shoulder_dist = calculate_distance(right_shoulder, right_wrist)
    left_hand_to_shoulder_dist = calculate_distance(left_shoulder, left_wrist)

    # Define thresholds for "outward" hand movement based on some empirical values
    movement_threshold = 0.1  # Adjust based on your testing

    # Check if hands are moving outward beyond a certain threshold from the shoulders
    if right_hand_to_shoulder_dist > movement_threshold and left_hand_to_shoulder_dist > movement_threshold:
        # You could also add a check for rhythmic movement here (e.g., detect repetition over frames)
        return True

    return False

# Video capture from IP camera or webcam
cap = cv2.VideoCapture('http://192.168.29.3:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Export coordinates
        try:
            # Extract Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face Landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten())

            # Concat Rows
            row = pose_row + face_row

            # Check for "bird wings" gesture
            if results.right_hand_landmarks and results.left_hand_landmarks and results.pose_landmarks:
                if is_bird_wings(results.right_hand_landmarks.landmark, results.left_hand_landmarks.landmark, results.pose_landmarks.landmark):
                    print("Bird Wings Gesture Detected!")
                    row.insert(0, "Hand Stims (Bird Wings)")
                else:
                    row.insert(0, class_name)
            else:
                row.insert(0, class_name)

            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            pass
        
        # Display the image
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources after loop ends
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[ ]:





# In[28]:


# Check if a significant number of landmarks are missing (e.g., less than 384 filled landmarks)
if num_coords >= 384:
    # Proceed with exporting or processing
    print(f"Processing {num_coords} landmarks")
else:
    print(f"Skipping frame with only {num_coords} landmarks detected")


# # Step-3 : Train Custom Model Using Scikit Learn

# # Step-3.1 : Read in Collected Data and Process

# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[32]:


df = pd.read_csv('coords.csv')


# In[34]:


df.head()


# In[36]:


df.tail()


# In[156]:


#df[df['class']=='Happy']


# In[158]:


#df[df['class']=='Sad']


# In[160]:


#df[df['class']=='Victorious']


# In[162]:


#df[df['class']=='RigidPosture']


# In[164]:


#df[df['class']=='SpiderFingers']


# In[166]:


#df[df['class']=='HandTrumpet']


# In[168]:


#df[df['class']=='Hand Stims (Hand Trumpet)']


# In[170]:


#df[df['class']=='BirdWings']


# In[172]:


#df[df['class']=='Hand Stims (Bird Wings)']


# In[48]:


x = df.drop('class', axis=1) # Features
y = df['class'] # Target Value


# In[50]:


# Step 1: Select the first 384 landmarks as features
x = df.iloc[:, :384]  # Select the first 384 columns for landmark coordinates

# Step 2: Select the 'class' column as the target
y = df['class']  # Select the column with action labels


# In[52]:


x


# In[54]:


y


# In[56]:


from sklearn.model_selection import train_test_split

# Step 3: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)


# In[58]:


x_train


# In[60]:


y_train


# In[62]:


x_test


# In[64]:


y_test


# # Step-3.2 : Train Machine Learning Classification Model

# In[66]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[68]:


pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), # Increased max_iter to avoid convergence issue
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}


# In[70]:


list(pipelines.values())[0]


# In[72]:


pipelines.keys()


# In[74]:


# Print the data types of each column
print(x_train.dtypes)


# In[76]:


# Print the data types of each column
print(y_train.dtypes)


# In[78]:


# Print the data types of each column
print(x_test.dtypes)


# In[80]:


# Print the data types of each column
print(y_test.dtypes)


# In[82]:


# Filter out columns that are non-numeric
non_numeric_cols = x_train.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_cols)


# In[ ]:


#fit_models = {}
#for algo, pipeline in pipelines.items():
    #model = pipeline.fit(x_train, y_train)
    #fit_models[algo] = model


# In[84]:


from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Define categorical and numeric columns
categorical_cols = ['class']  # Only the 'class' column is categorical
numeric_cols = x_train.select_dtypes(exclude=['object']).columns.tolist()  # Automatically select numeric columns

# Define the preprocessing steps for numeric and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
            ('scaler', StandardScaler())  # Standardize numeric features
        ]), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)  # OneHotEncode categorical features
    ])

fit_models = {}

# Modify your pipelines to include the preprocessor without duplicating the name
for algo, pipeline in pipelines.items():
    # Create a new pipeline that includes the preprocessor and the existing model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', pipeline.steps[-1][1])  # Assuming the last step is the model
    ])
    fit_models[algo] = model.fit(x_train, y_train)


# In[85]:


fit_models


# In[88]:


fit_models['lr']


# In[90]:


fit_models['rc']


# In[92]:


fit_models['rf']


# In[94]:


fit_models['gb']


# In[96]:


fit_models['lr'].predict(x_test)


# In[98]:


fit_models['rc'].predict(x_test)


# In[100]:


fit_models['rf'].predict(x_test)


# In[102]:


fit_models['gb'].predict(x_test)


# # Step-3.2.1 : Data Validation Check And Rectifications

# In[104]:


y_test.value_counts()


# In[106]:


from sklearn.metrics import classification_report, confusion_matrix

for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(f"Model: {algo}")
    print(classification_report(y_test, yhat))
    print(confusion_matrix(y_test, yhat))


# In[108]:


print(x.shape)
print(y.shape)


# In[110]:


print(x.shape)
print(x.size)  # Total number of elements


# In[112]:


# Example: Load the full dataset instead of just one sample
x = pd.read_csv('coords.csv')  # Or whatever method you use
print(x.shape)  # Should return (509, 1536) or something close


# In[114]:


x_array = x.iloc[:, :1536].values  # Select first 1536 columns


# In[116]:


non_numeric_columns = x.select_dtypes(include=['object']).columns
print(non_numeric_columns)  # To see the non-numeric columns


# In[118]:


from sklearn.preprocessing import LabelEncoder

# Apply LabelEncoder to non-numeric columns
for col in non_numeric_columns:
    x[col] = LabelEncoder().fit_transform(x[col])


# In[120]:


from sklearn.impute import SimpleImputer

# Impute missing values (replace NaN with column mean)
imputer = SimpleImputer(strategy='mean')  # You can use 'median', 'most_frequent', or 'constant'
x_imputed = imputer.fit_transform(x)

# Now apply PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=1536)
x_reduced = pca.fit_transform(x_imputed)


# In[124]:


# Check the shape of x
print(x.shape)  # This will give you the number of rows and columns


# In[126]:


from sklearn.impute import SimpleImputer

# Create an imputer that fills NaNs with the mean of each column
imputer = SimpleImputer(strategy='mean')
x_imputed = imputer.fit_transform(x.values)

# Proceed with PCA
pca = PCA(n_components=min(509, 2005))
x_reduced = pca.fit_transform(x_imputed)

print(x_reduced.shape)  # Check the shape after PCA


# In[128]:


from sklearn.model_selection import cross_val_score

for algo, model in fit_models.items():
    scores = cross_val_score(model, x, y, cv=5)  # 5-fold cross-validation
    print(f"Model: {algo}, Mean Accuracy: {scores.mean()}")


# # Step-3.3 : Evaluate and Serialize Model

# In[130]:


from sklearn.metrics import accuracy_score # Accuracy Metrics
import pickle


# In[132]:


for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))


# In[134]:


fit_models['rf'].predict(x_test)


# In[136]:


y_test


# In[138]:


with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rc'],f)


# # Step-4 : Make Detections With Model

# In[140]:


with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


# In[142]:


model


# In[ ]:


get_ipython().system('pip install protobuf==3.20.3')

# !pip install joblib
# import joblib  # or use pickle

# try:
#     model = joblib.load(r'C:\Users\lakshana V\body_language.pkl')
# except Exception as e:
#     print(f"Error loading model: {e}")

import pickle  # instead of joblib

# Load the model using pickle
try:
    with open(r'C:\Users\lakshana V\body_language.pkl', 'rb') as f:  # open in 'rb' (read binary) mode
        model = pickle.load(f)
    print("Model loaded successfully with pickle")
except Exception as e:
    print(f"Error loading model with pickle: {e}")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:

    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

         # Export coordinates
        try:
            # Extract Pose Landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract Face Landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in face]).flatten())

            # Concat Rows
            row = pose_row+face_row

            # # Append Class Name
            # row.insert(0, class_name)
            
            # # Export to CSV
            # with open('coords.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row)

            # Make Detections
            x = pd.DataFrame([row])
            print("Data for prediction:", x)
            body_language_class = model.predict(x)[0]
            body_language_prob = model.predict_proba(x)[0]
            print(body_language_class, body_language_prob)

            # If you're printing predictions in a loop, limit it
            if i == desired_iteration:  # Specify a condition to print only certain outputs
                print(prediction)

            print(prediction.iloc[:, :10])  # Print only the first 10 columns of the prediction
            print(prediction.round(2))  # Round all predictions to 2 decimal places
            print(prediction.mean())  # Display the mean of each feature across all rows

            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
                            ), [1920,1080]).astype(int))

            cv2.rectangle(image, (coords[0], coords[1]+5), 
                           (coords[0]+len(body_language_class)*20, coords[1]-30),
                           (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords,
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        

            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass
        
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

# Message after the webcam feed is closed
print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# In[328]:


results.pose_landmarks.landmark


# In[330]:


results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]


# In[332]:


np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [1920,1080])         


# In[ ]:





# # Step-5 : Cells For Final Run

# # Install and Imports

# In[144]:


# Install the required protobuf version
get_ipython().system('pip install protobuf==3.20.3')

# Import necessary libraries
import cv2
import csv
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle
import warnings
import collections

# Filter specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')


# # Load The Model 

# In[146]:


try:
    with open(r'C:\Users\lakshana V\body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully with pickle")
except Exception as e:
    print(f"Error loading model with pickle: {e}")


# # Initialize Mediapipe

# In[148]:


# Initialize MediaPipe holistic and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# # Thresholds and Variables

# In[160]:


import collections  # Import collections module

# Function to detect Rigid Posture
# Threshold for rigid posture detection
rigid_threshold = 0.01  # Adjust this value as needed
# Variables to store the previous frame's pose landmarks for rigid posture comparison
prev_pose_landmarks = None

# Buffer for rigid posture smoothing
buffer_size = 10
rigid_posture_buffer = collections.deque(maxlen=buffer_size)

# Function to check if the majority of the frames classify the posture as rigid
def is_rigid_posture(buffer):
    return sum(buffer) > len(buffer) / 2

# Threshold for rigid posture detection
rigid_threshold = 0.01  # Adjust this value as needed
# Variables to store the previous frame's pose landmarks for rigid posture comparison
prev_pose_landmarks = None

# Buffer for rigid posture smoothing
buffer_size = 10
rigid_posture_buffer = collections.deque(maxlen=buffer_size)

# Function to check if the majority of the frames classify the posture as rigid
def is_rigid_posture(buffer):
    return sum(buffer) > len(buffer) / 2

# Function to detect Spider Fingers
def detect_spider_fingers(hand_landmarks):
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        # Calculate distances between the thumb tip and each finger tip
        dist_thumb_index = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        dist_thumb_middle = np.sqrt((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2)
        dist_thumb_ring = np.sqrt((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2)
        dist_thumb_pinky = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)

        # Check if the fingers are spread out and slightly curved (conditions for Spider Fingers)
        if (dist_thumb_index > 0.15 and dist_thumb_middle > 0.15 and 
            dist_thumb_ring > 0.15 and dist_thumb_pinky > 0.15):
            return True
    return False

# Function to detect hand trumpet
def is_hand_trumpet(right_hand_landmarks, face_landmarks):
    if right_hand_landmarks and face_landmarks:
        # Check distance between hand and face
        hand_to_face_distance = np.sqrt(
            (right_hand_landmarks.landmark[0].x - face_landmarks.landmark[1].x) ** 2 + 
            (right_hand_landmarks.landmark[0].y - face_landmarks.landmark[1].y) ** 2
        )
         # Define thresholds for hand being near the face and fingers being close together
        hand_near_face_threshold = 0.1  # Adjust as needed
        thumb_index_distance = np.sqrt(
            (right_hand_landmarks.landmark[4].x - right_hand_landmarks.landmark[8].x) ** 2 +
            (right_hand_landmarks.landmark[4].y - right_hand_landmarks.landmark[8].y) ** 2
        )
        thumb_index_cup_threshold = 0.05  # Adjust as needed

        # If the hand is close to the face and the thumb and index are close together, detect hand trumpet
        if hand_to_face_distance < hand_near_face_threshold and thumb_index_distance < thumb_index_cup_threshold:
            return True
    
    return False

# Function to detect Bird Wings gesture
def detect_bird_wings(pose_landmarks):
    if pose_landmarks:
        # Define landmarks for shoulders and wrists
        left_shoulder = pose_landmarks.landmark[11]  # Left shoulder
        right_shoulder = pose_landmarks.landmark[12]  # Right shoulder
        left_wrist = pose_landmarks.landmark[15]      # Left wrist
        right_wrist = pose_landmarks.landmark[16]     # Right wrist

        # Calculate distances between shoulders and wrists
        shoulder_distance = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
        wrist_distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)

        # Define thresholds for Bird Wings gesture
        shoulder_threshold = 0.3  # Shoulders should be spread out
        wrist_height_threshold = 0.2  # Wrists should be raised above the shoulder height

        # Check for Bird Wings conditions
        if (shoulder_distance > shoulder_threshold and 
            (left_wrist.y < left_shoulder.y - wrist_height_threshold) and 
            (right_wrist.y < right_shoulder.y - wrist_height_threshold)):
            return True

    return False


# # Detection Logic

# In[162]:


import collections  # Import collections module
import numpy as np  # Make sure to import numpy if you haven't already

# Threshold for rigid posture detection
rigid_threshold = 0.01  # Adjust this value as needed

# Buffer for rigid posture smoothing
buffer_size = 10
rigid_posture_buffer = collections.deque(maxlen=buffer_size)

def is_rigid_posture(buffer):
    """
    Check if the majority of the frames classify the posture as rigid.
    """
    return sum(buffer) > len(buffer) / 2

def detect_rigid_posture(current_pose_landmarks, prev_pose_landmarks):
    """
    Logic to detect Rigid Posture by comparing current and previous pose landmarks.
    """
    if prev_pose_landmarks is None:
        return False  # Can't compare if there is no previous frame

    # Compute the difference between the current and previous landmarks
    differences = [abs(current_pose_landmarks[i].y - prev_pose_landmarks[i].y) for i in range(len(current_pose_landmarks))]

    # Check if all differences are below the rigid posture threshold
    is_rigid = all(diff < rigid_threshold for diff in differences)

    # Update the buffer for smoothing
    rigid_posture_buffer.append(1 if is_rigid else 0)

    # Return whether the current posture is rigid based on buffer
    return is_rigid_posture(rigid_posture_buffer)

def detect_spider_fingers(hand_landmarks):
    """
    Logic to detect Spider Fingers gesture by analyzing the relative positions
    of the finger landmarks.
    """
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        # Calculate distances between the thumb tip and each finger tip
        dist_thumb_index = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        dist_thumb_middle = np.sqrt((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2)
        dist_thumb_ring = np.sqrt((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2)
        dist_thumb_pinky = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)

        # Check if the fingers are spread out and slightly curved (conditions for Spider Fingers)
        if (dist_thumb_index > 0.15 and dist_thumb_middle > 0.15 and 
            dist_thumb_ring > 0.15 and dist_thumb_pinky > 0.15):
            return True
    return False

def detect_hand_trumpet(hand_landmarks, face_landmarks):
    """
    Logic to detect Hand Trumpet gesture by analyzing the relative positions
    of the thumb and fingers in relation to the face.
    """
    if hand_landmarks and face_landmarks:  # Check if both landmarks are available
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]

        # Ensure that the landmarks are available before accessing them
        if len(face_landmarks.landmark) > 1:  # Check if there are enough landmarks
            face_position = face_landmarks.landmark[1]  # Assuming the face landmark to check is the nose
            hand_to_face_distance = np.sqrt(
                (thumb_tip.x - face_position.x) ** 2 + 
                (thumb_tip.y - face_position.y) ** 2
            )

            # Logic: If thumb is extended and fingers are curved towards it
            if (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and
                hand_to_face_distance < 0.1):  # Adjust distance threshold as necessary
                return True
            
    return False

def detect_bird_wings(hand_landmarks):
    """
    Logic to detect Bird Wings gesture by analyzing the relative positions
    of the hand landmarks.
    """
    if hand_landmarks:
        shoulder_left = hand_landmarks.landmark[11]
        shoulder_right = hand_landmarks.landmark[12]
        wrist_left = hand_landmarks.landmark[9]
        wrist_right = hand_landmarks.landmark[10]

        # Logic: If wrists are significantly lower than shoulders, it indicates a bird wings position
        if (wrist_left.y > shoulder_left.y) and (wrist_right.y > shoulder_right.y):
            return True
    return False


# # Run the Holistic Model and Apply Gesture Detection

# In[181]:


# Capture video from webcam (or IP camera)
cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3) as holistic:
    print("\n🌟 Your Empathion is in connection! Let's make incredible strides together in decoding body language! 🌟")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Initialize variables for class label and probability
        body_language_class = "Unknown"
        body_language_prob = [0]

        # Check for rigid posture
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark

            if prev_pose_landmarks:
                rigid_posture = True  # Assume rigid posture by default
                for idx in [11, 12, 23, 24, 25, 26]:
                    prev_landmark = prev_pose_landmarks[idx]
                    curr_landmark = pose_landmarks[idx]

                    movement = np.sqrt((curr_landmark.x - prev_landmark.x) ** 2 + (curr_landmark.y - prev_landmark.y) ** 2)

                    if movement > rigid_threshold:
                        rigid_posture = False
                        break

                rigid_posture_buffer.append(1 if rigid_posture else 0)

                if is_rigid_posture(rigid_posture_buffer):
                    body_language_class = "RigidPosture"
                    body_language_prob = [1.0]

            prev_pose_landmarks = pose_landmarks

        # Check for Spider Fingers
        if detect_spider_fingers(results.right_hand_landmarks) or detect_spider_fingers(results.left_hand_landmarks):
            body_language_class = "SpiderFingers"
            body_language_prob = [1.0]

        # Check for Hand Trumpet
        if (detect_hand_trumpet(results.right_hand_landmarks, results.face_landmarks) or detect_hand_trumpet(results.left_hand_landmarks, results.face_landmarks)):
            body_language_class = "HandTrumpet"
            body_language_prob = [1.0]

        # Check for Bird Wings
        if detect_bird_wings(results.right_hand_landmarks) or detect_bird_wings(results.left_hand_landmarks):
            body_language_class = "BirdWings"
            body_language_prob = [1.0]

        if body_language_class not in ["RigidPosture", "SpiderFingers", "HandTrumpet", "BirdWings"]:
            try:
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten())
                face_row = list(np.array([[landmark.x, landmark.y, landmark.visibility] for landmark in results.face_landmarks.landmark]).flatten())
                row = pose_row + face_row

                x = pd.DataFrame([row])
                body_language_class = model.predict(x)[0]
                body_language_prob = model.predict_proba(x)[0]
            except:
                pass

        # Display class and probability on the screen
        text = f"Class: {body_language_class}"
        prob_text = f"Prob: {round(max(body_language_prob), 2)}"

        # Font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2
        bg_color = (128, 128, 0)

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        x, y = 10, 50

        # Draw background and text
        cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y + 5), bg_color, -1)
        cv2.putText(image, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(image, prob_text, (10, 80), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Empathized Kinetic Intel : Web-Cam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("🌿 Hope you enjoyed the magic happening on screen! 🌿")


# # --- THE END ---
