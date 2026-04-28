import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import math
import mediapipe as mp


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)

    def extract_landmarks_dict(self, results):
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark

        def get(i):
            p = lm[i]
            return (p.x, p.y, p.visibility)

        return {
            "nose": get(0),
            "left_eye": get(2),
            "right_eye": get(5),
            "left_ear": get(7),
            "right_ear": get(8),
            "left_shoulder": get(11),
            "right_shoulder": get(12),
            "left_elbow": get(13),
            "right_elbow": get(14),
            "left_wrist": get(15),
            "right_wrist": get(16),
            "left_hip": get(23),
            "right_hip": get(24),
            "left_knee": get(25),
            "right_knee": get(26),
            "left_ankle": get(27),
            "right_ankle": get(28),
            "left_heel": get(29),
            "right_heel": get(30),
            "left_foot_index": get(31),
            "right_foot_index": get(32),
        }

    def draw_landmarks_on_frame(self, frame, results):
        annotated = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated

    def close(self):
        self.pose.close()


def calculate_angle_degrees(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
