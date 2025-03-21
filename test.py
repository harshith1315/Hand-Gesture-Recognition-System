import cv2
import mediapipe as mp
import time
import sys
import numpy as np
from tensorflow.keras.models import load_model


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        if img is None:
            return None
            
        # Store image dimensions
        self.image_height, self.image_width, _ = img.shape
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            # Create NormalizedRect with image dimensions
            self.results = self.hands.process(img_rgb)
            
            # Update MediaPipe's internal image dimensions
            if hasattr(self.hands, '_image_height'):
                self.hands._image_height = self.image_height
            if hasattr(self.hands, '_image_width'):
                self.hands._image_width = self.image_width
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return img
            
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    try:
                        self.mp_draw.draw_landmarks(
                            img, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                    except Exception as e:
                        print(f"Error drawing landmarks: {e}")
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        landmark_list = []
        if img is None:
            return landmark_list
            
        if hasattr(self, 'results') and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                try:
                    hand = self.results.multi_hand_landmarks[hand_no]
                    height, width, _ = img.shape
                    for id, landmark in enumerate(hand.landmark):
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        landmark_list.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
        return landmark_list

class GestureRecognizer:
    def __init__(self, model_path='gesture_model.h5'):
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully")
            # self.model.load_weights('gesture_model_weights.h5')
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
            
    def preprocess_landmarks(self, landmark_list, img_shape=(256, 256)):
        if len(landmark_list) == 0:
            return None
            
        # Create a blank image
        gesture_img = np.zeros(img_shape, dtype=np.uint8)
        
        # Get bounding box of hand landmarks
        x_coords = [lm[1] for lm in landmark_list]
        y_coords = [lm[2] for lm in landmark_list]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img_shape[1], max_x + padding)
        max_y = min(img_shape[0], max_y + padding)
        
        # Draw landmarks and connections on blank image
        for i in range(len(landmark_list) - 1):
            start_point = (landmark_list[i][1], landmark_list[i][2])
            end_point = (landmark_list[i + 1][1], landmark_list[i + 1][2])
            cv2.line(gesture_img, start_point, end_point, 255, 2)
            cv2.circle(gesture_img, start_point, 3, 255, -1)
        
        # Draw the last point
        last_point = (landmark_list[-1][1], landmark_list[-1][2])
        cv2.circle(gesture_img, last_point, 3, 255, -1)
        
        # Crop and resize the hand region
        hand_region = gesture_img[min_y:max_y, min_x:max_x]
        if hand_region.size == 0:
            return None
            
        resized_img = cv2.resize(hand_region, img_shape)
        
        # Normalize and reshape for model input
        processed_img = resized_img.astype(np.float32) / 255.0
        processed_img = np.expand_dims(processed_img, axis=-1)  # Add channel dimension
        processed_img = np.expand_dims(processed_img, axis=0)   # Add batch dimension
        
        return processed_img
        
    def predict_gesture(self, landmark_list):
        processed_data = self.preprocess_landmarks(landmark_list)
        if processed_data is None:
            return None
            
        try:
            prediction = self.model.predict(processed_data, verbose=0)
            return np.argmax(prediction[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize detector and gesture recognizer
    detector = HandDetector()
    gesture_recognizer = GestureRecognizer()
    
    # Initialize FPS calculation
    prev_time = 0
    fps = 0
    
    # Dictionary to map prediction indices to gesture names
    # Update this according to your model's classes
    gesture_dict = {
        1: "hii", 2: "fist", 3: "c", 4: "rad", 5: "E",
        6: "peace", 7: "fist", 8: "okey"
        # Add more gesture mappings based on your model
    }
    
    try:
        while True:
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries:
                success, img = cap.read()
                if success and img is not None:
                    break
                print(f"Failed to grab frame, attempt {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(0.1)
            
            if not success or img is None:
                print("Failed to grab frame after multiple attempts, restarting capture")
                cap.release()
                cap = cv2.VideoCapture(0)
                continue
                
            # Find hands and landmarks
            img = detector.find_hands(img)
            landmark_list = detector.find_position(img)
            
            # Predict gesture if hand is detected
            if len(landmark_list) > 0:
                gesture_id = gesture_recognizer.predict_gesture(landmark_list)
                if gesture_id is not None:
                    gesture = gesture_dict.get(gesture_id, "Unknown")
                    
                    # Display gesture prediction
                    cv2.putText(img, f'Gesture: {gesture}', (10, 130), 
                              cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            
            # Calculate and display FPS
            current_time = time.time()
            if prev_time > 0:
                fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            try:
                cv2.putText(img, f'FPS: {int(fps)}', (10, 70), 
                           cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                
                cv2.imshow("Hand Gesture Recognition", img)
            except Exception as e:
                print(f"Error displaying frame: {e}")
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting program")
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    