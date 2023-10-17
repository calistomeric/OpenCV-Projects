import cv2
import argparse
import os

def detect_face(filename, webcam):
    cam = cv2.VideoCapture(webcam)
    cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), f"dataset/{filename}"))
    while True:
        # reading image from webcam
        ret, image = cam.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cascade.detectMultiScale(gray_image, 1.3, 5)
        for x, y, w, h in face_cascade:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0),3)
        key = cv2.waitKey(1)
        cv2.imshow('Face Detection', image)
        if key == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detects face of a person")
    parser.add_argument("--filename", type=str, default = "haarcascade_frontalface_default.xml", 
                        help="Name of the opencv  model for face detection. Usually in .xml format")
    parser.add_argument("--webcam", type=int, default = 0, 
                        help="Number that activates your camera. Usually 0 for most people")
    args = parser.parse_args()
    detect_face(args.filename, args.webcam)
    
    
                
        
    
