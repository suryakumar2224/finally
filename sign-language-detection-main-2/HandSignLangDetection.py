import cv2
import mediapipe as mp

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Hand landmark indices for fingertips and thumb tip
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# Hand Sign Detection Functions
def check_thank_you(lm_list):
    # 'Thank You' sign: Fingers extended, palm facing outward
    if lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y:
        return True
    return False

def check_i_love_you(lm_list):
    # 'I Love You' sign: Pinky and thumb extended, other fingers folded
    if lm_list[4].y < lm_list[3].y and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
        return True
    return False

def check_ok_sign(lm_list):
    # 'OK' sign: Thumb touches index finger, other fingers extended
    if lm_list[4].x < lm_list[8].x and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y:
        return True
    return False

def check_rock_sign(lm_list):
    # 'Rock' sign: Index and pinky fingers up, others folded
    if lm_list[8].y < lm_list[6].y and lm_list[20].y < lm_list[18].y and lm_list[12].y > lm_list[10].y:
        return True
    return False

def check_call_me_sign(lm_list):
    # 'Call Me' sign: Thumb and pinky outstretched, others folded
    if lm_list[4].x < lm_list[3].x and lm_list[20].y < lm_list[18].y and lm_list[8].y > lm_list[6].y:
        return True
    return False

def check_like_sign(lm_list):
    # 'Like' sign: Thumb up, other fingers folded
    if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y and all(lm_list[i].x < lm_list[i-2].x for i in finger_tips):
        return True
    return False

def check_dislike_sign(lm_list):
    # 'Dislike' sign: Thumb down, other fingers folded
    if lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y and all(lm_list[i].x < lm_list[i-2].x for i in finger_tips):
        return True
    return False

def check_victory_sign(lm_list):
    # 'Victory' sign: Index and middle fingers up, others folded
    if lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
        return True
    return False

def check_peace_sign(lm_list):
    # 'Peace' sign: Index and middle fingers up, thumb to the side, ring and pinky down
    if lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
        return True
    return False

def check_good_morning(lm_list):
    # 'Good Morning' sign: Hand waved with fingers extended (simplified version)
    if lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y:
        return True
    return False

def check_smile_sign(lm_list):
    # 'Smile' sign: All fingers extended, palm facing up
    if all(lm_list[tip].y < lm_list[tip - 2].y for tip in finger_tips) and lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y:
        return True
    return False

# Main Loop
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Check for hand signs
            if check_thank_you(lm_list):
                cv2.putText(img, "THANK YOU", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            elif check_i_love_you(lm_list):
                cv2.putText(img, "I LOVE YOU", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            elif check_ok_sign(lm_list):
                cv2.putText(img, "OK", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 3)
            elif check_rock_sign(lm_list):
                cv2.putText(img, "ROCK", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 3)
            elif check_call_me_sign(lm_list):
                cv2.putText(img, "CALL ME", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 192, 203), 3)
            elif check_like_sign(lm_list):
                cv2.putText(img, "LIKE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif check_dislike_sign(lm_list):
                cv2.putText(img, "DISLIKE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif check_victory_sign(lm_list):
                cv2.putText(img, "VICTORY", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            elif check_peace_sign(lm_list):
                cv2.putText(img, "PEACE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif check_good_morning(lm_list):
                cv2.putText(img, "GOOD MORNING", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 3)
            elif check_smile_sign(lm_list):
                cv2.putText(img, "SMILE", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 192, 203), 3)

            # Draw landmarks on the hand
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    # Show the output image
    cv2.imshow("Hand Sign Detection", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
