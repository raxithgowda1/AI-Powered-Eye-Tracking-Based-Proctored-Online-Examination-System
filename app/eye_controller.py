import cv2
import dlib
import numpy as np
import pyautogui
from imutils import face_utils
from utils import eye_aspect_ratio, mouth_aspect_ratio, direction
import time

class EyeController:
    def __init__(self):
        self.active = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        self.eye_ar_thresh = 0.19
        self.mouth_ar_thresh = 0.3
        self.input_mode = False
        self.scroll_mode = False
        self.anchor_point = (0, 0)
        self.last_blink_time = time.time()
        self.last_mouth_time = time.time()
        self.last_left_click_time = time.time()
        self.last_right_click_time = time.time()
        self.cooldown_period = 1.0  
        self.click_cooldown = 0.5  
        
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1

    def activate(self):
        self.active = True
        self.input_mode = False
        self.scroll_mode = False
        print("Eye controller activated")

    def deactivate(self):
        self.active = False
        self.input_mode = False
        self.scroll_mode = False
        print("Eye controller deactivated")

    def process_frame(self, frame):
        if not self.active:
            return frame

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        current_time = time.time()

        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            mouth = shape[48:68]
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            nose = shape[27:36]

            mar = mouth_aspect_ratio(mouth)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            nose_point = (nose[3, 0], nose[3, 1])

            if mar > self.mouth_ar_thresh and current_time - self.last_mouth_time > self.cooldown_period:
                self.input_mode = not self.input_mode
                self.last_mouth_time = current_time
                if self.input_mode:
                    self.anchor_point = nose_point
                    print("Input mode activated")
                else:
                    print("Input mode deactivated")

            if ear < self.eye_ar_thresh and current_time - self.last_blink_time > self.cooldown_period:
                self.scroll_mode = not self.scroll_mode
                self.last_blink_time = current_time
                if self.scroll_mode:
                    print("Scroll mode activated")
                else:
                    print("Scroll mode deactivated")
            
            if left_ear < self.eye_ar_thresh and right_ear > self.eye_ar_thresh and current_time - self.last_left_click_time > self.click_cooldown:
                if self.input_mode:  
                    pyautogui.click(button='left')
                    print("Left click performed")
                    self.last_left_click_time = current_time
                    
                    cv2.putText(frame, "LEFT CLICK!", (int(frame.shape[1]/2) - 60, int(frame.shape[0]/2)),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            if right_ear < self.eye_ar_thresh and left_ear > self.eye_ar_thresh and current_time - self.last_right_click_time > self.click_cooldown:
                if self.input_mode:  
                    pyautogui.click(button='right')
                    print("Right click performed")
                    self.last_right_click_time = current_time
                    
                    cv2.putText(frame, "RIGHT CLICK!", (int(frame.shape[1]/2) - 60, int(frame.shape[0]/2)),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if self.input_mode:
                try:
                    dir = direction(nose_point, self.anchor_point, 30, 15) 
                    self.handle_movement(dir)
                except Exception as e:
                    print(f"Movement error: {e}")

            for (x, y) in np.concatenate((mouth, left_eye, right_eye), axis=0):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            self.draw_indicators(frame, self.input_mode, self.scroll_mode, left_ear, right_ear, mar)
            
            if self.input_mode:
                cv2.putText(frame, "INPUT MODE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.scroll_mode:
                cv2.putText(frame, "SCROLL MODE", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(frame, "EYE CONTROL MODE", (frame.shape[1] - 230, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.input_mode:
                cv2.putText(frame, "Close left eye: left click", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, "Close right eye: right click", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame

    def draw_indicators(self, frame, input_mode, scroll_mode, left_ear, right_ear, mar):
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (frame.shape[1] - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (frame.shape[1] - 150, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(frame, f"MAR: {mar:.2f}", (frame.shape[1] - 150, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        color_input = (0, 255, 0) if input_mode else (0, 0, 255)
        color_scroll = (0, 255, 0) if scroll_mode else (0, 0, 255)
        
        cv2.circle(frame, (frame.shape[1] - 20, 30), 10, color_input, -1)
        cv2.circle(frame, (frame.shape[1] - 20, 60), 10, color_scroll, -1)

    def handle_movement(self, direction):
        drag = 10  
        
        try:
            if direction == 'right':
                pyautogui.moveRel(drag, 0)
            elif direction == 'left':
                pyautogui.moveRel(-drag, 0)
            elif direction == 'up':
                if self.scroll_mode:
                    pyautogui.scroll(30)
                else:
                    pyautogui.moveRel(0, -drag)
            elif direction == 'down':
                if self.scroll_mode:
                    pyautogui.scroll(-30)
                else:
                    pyautogui.moveRel(0, drag)
            elif direction == 'none':
                pass
        except Exception as e:
            print(f"PyAutoGUI error: {e}")