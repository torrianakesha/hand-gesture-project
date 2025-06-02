import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#HELPER CLASS FOR CAPTURING THE DATA
class DataHandCapture:

    def __init__(self, file_name: str):
        self.file_name = file_name

    def make_header(self, save=True):
        header = []

        for i in range(21):
            header.append(f"x{i}-1")
            header.append(f"y{i}-1")
            header.append(f"z{i}-1")

        if save:
            header.append("label")
            with open(self.file_name, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)
        else:
            return header

    def set_results(self, hand_results):
        self.hand_results = hand_results

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x <= 120 and 420 <= y <= 470:
                with open(self.file_name, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)

                    lm_row = []

                    for hand_landmarks in self.hand_results.multi_hand_landmarks:
                        
                        for index, lm in enumerate(mp_hands.HandLandmark):
                            lm_point = hand_landmarks.landmark[index]
                            lm_row.append(lm_point.x)
                            lm_row.append(lm_point.y)
                            lm_row.append(lm_point.z)
                        
                    csv_writer.writerow(lm_row)
                    


# MAIN FUNCTION
def main():

    window_name = "My Camera"
    file_name ="trainingdata.csv"

    hc = DataHandCapture(file_name=file_name)

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        hc.make_header()

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            success, image = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Default bounding box color set to red
            cv2.rectangle(image, (90,70),(520,400), (0, 0, 255), 2)
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image)
            hc.set_results(results)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                max_x = 0
                min_x = 1
                max_y = 0
                min_y = 1

                for hand_landmarks in results.multi_hand_landmarks:
                    
                    for point in hand_landmarks.landmark:
                        max_x = max(max_x,point.x)
                        min_x = min(min_x,point.x)
                        max_y = max(max_y,point.y)
                        min_y = min(min_y,point.y)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                 # Condition for box color for hands
                if max_x < 0.80 and min_x > 0.15 and max_y < 0.82 and min_y > 0.18:
                    text = "Capture"
                    # Change the bounding box color to green when the hand is inside it
                    cv2.rectangle(image, (0,420),(120,470), (0,255,0), cv2.FILLED)
                    cv2.putText(image, "Capture", (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
                    cv2.rectangle(image, (90,70),(520,400), (0, 255, 0), 2)
                    cv2.setMouseCallback(window_name, hc.handle_click)

            cv2.imshow(window_name, image)
            if cv2.waitKey(5) & 0xFF == 27:
                break


if __name__ == "__main__":
    main()


