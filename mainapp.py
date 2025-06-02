from training import trainedmodel # Importing csv file of the trained model
import cv2
import mediapipe as mp
import os
import pandas as pd
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def main():

    window_name = "My Camera"
    presentationPath = "file"
    imgNum = 4 # number of slides
    btnPressed = False # default value of btn for 
    counter = 0 # button counter
    delay = 5 # button delay, 30
    delay_counter = 0
    delay_threshold = 5  # Adjust this threshold based on your requirements

    # List of Images for Presentation
    pathImages = sorted(os.listdir(presentationPath), key=len)
    print(pathImages)
        
    # Webcam 
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
            
        # 
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        pathFullImage = os.path.join(presentationPath, pathImages[imgNum])  # Presentation image path
        imgCurrent = cv2.imread(pathFullImage)    

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            success, image = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Default Color Red
            cv2.rectangle(image, (90,70),(520,400), (0, 0, 255), 2)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            hand_inside = False
            
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is detected
                
                max_x = max([lm.x for lm in hand_landmarks.landmark])
                min_x = min([lm.x for lm in hand_landmarks.landmark])
                max_y = max([lm.y for lm in hand_landmarks.landmark])
                min_y = min([lm.y for lm in hand_landmarks.landmark])
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Check if the hand is inside the bounding box
                if max_x < 0.80 and min_x > 0.15 and max_y < 0.82 and min_y > 0.18:
                    hand_inside = True
                    cv2.rectangle(image, (90,70),(520,400), (0, 255, 0), 2)
                    
                else: 
                    hand_inside = False
            
                if hand_inside:
                    delay_counter += 1
                    if delay_counter > delay_threshold:
                        delay_counter = 0  # Reset the counter
                    
                    # Process gestures only when the hand is inside the bounding box
                    data = {}
                    for index, lm in enumerate(mp_hands.HandLandmark):
                        lm_point = hand_landmarks.landmark[index]
                        data[f"x{index}-1"] = [lm_point.x]
                        data[f"y{index}-1"] = [lm_point.y]
                        data[f"z{index}-1"] = [lm_point.z]
                        
                    df = pd.DataFrame(data)
                    tm = trainedmodel().predict(df)
                    
                    if tm[0] == "Previous" and not btnPressed:
                        if imgNum > 0:
                            btnPressed = True
                            imgNum -= 1
                            print("Previous")


                    elif tm[0] == "Next" and not btnPressed:
                        if imgNum < len(pathImages) - 1:
                            btnPressed = True
                            imgNum += 1
                            print("Next")
            else:
                pass # Do nothing if hand is not inside the bounding box

                if btnPressed:
                    counter += 1
                    if counter > delay:
                        counter = 0
                        btnPressed = False


            cv2.imshow(window_name, image)
            pathFullImage = os.path.join(presentationPath, pathImages[imgNum])
            imgCurrent = cv2.imread(pathFullImage)
            cv2.imshow("Presentation", imgCurrent)
        
            if cv2.waitKey(5) & 0xFF == 27:
                break

if __name__ == "__main__":
    main()


            # if results.multi_hand_landmarks:
            #     max_x = 0
            #     min_x = 1
            #     max_y = 0
            #     min_y = 1

                # for hand_landmarks in results.multi_hand_landmarks:
                    
                #     for point in hand_landmarks.landmark:
                #         max_x = max(max_x,point.x)
                #         min_x = min(min_x,point.x)
                #         max_y = max(max_y,point.y)
                #         min_y = min(min_y,point.y)

                #     mp_drawing.draw_landmarks(
                #         image,
                #         hand_landmarks,
                #         mp_hands.HAND_CONNECTIONS,
                #         mp_drawing_styles.get_default_hand_landmarks_style(),
                #         mp_drawing_styles.get_default_hand_connections_style())
                
                # if max_x < 0.80 and min_x > 0.15 and max_y < 0.82 and min_y > 0.18:
                #     data = {}
                #     hand_count = 0
                #     for hand_landmarks in results.multi_hand_landmarks:
                #         hand_count += 1
                #         for index, lm in enumerate(mp_hands.HandLandmark):
                #             lm_point = hand_landmarks.landmark[index]
                #             data[f"x{index}-{hand_count}"] = [lm_point.x]
                #             data[f"y{index}-{hand_count}"] = [lm_point.y]
                #             data[f"z{index}-{hand_count}"] = [lm_point.z]
                            
                    #  ------

            #         if tm[0]==0 or tm[0] == 1: 
            #             cv2.rectangle(image, (0,420),(120,470), (0,255,0), cv2.FILLED)
            #             cv2.putText(image, str(tm[0]), (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
                    
            #          # Gesture 1 - Left
            #         if tm[0] == 0 and not btnPressed:
            #             if imgNum > 0:  # If not on the first slide
            #                 btnPressed = True
            #                 imgNum -= 1  # Go to previous slide

            #         # Gesture 2 - Right (assuming tm[0] == 1 is a right swipe)
            #         elif tm[0] == 1 and not btnPressed:
            #             if imgNum < len(pathImages) - 1:  # If not on the last slide
            #                 btnPressed = True
            #                 imgNum += 1  # Go to next slide

            #     else:
            #         btnPressed = False  # Reset button state if hand is outside the box

            # # Button Iterations
            # if btnPressed:
            #     counter += 1
            #     if counter > delay:
            #         counter = 0
            #         btnPressed = False

            # cv2.imshow(window_name, image)
            # cv2.imshow("Presentation", imgCurrent)
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break