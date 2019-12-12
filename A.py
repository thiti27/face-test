import cv2

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade=cv2.CascadeClassifier("haarcascade_eye.xml")

def create_dataset(img,id,img_id):
    # cv2.imwrite("../dataset/pic."+str(id)+"."+str(img_id)+".jpg",img)
    cv2.imwrite(+str(id)+"."+str(img_id)+".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        coords=[x,y,w,h]
    return img,coords

def detect(img, faceCascade,img_id):
    img,coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 0, 255), "Face")
    # create_dataset(img,id,img_id)

    return img


img_id = 0
try:
    cap = cv2.VideoCapture('rtsp://root:D@!cel16@192.168.1.3:554/live.sdp',)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while (True):
        ret,frame = cap.read()
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = detect(frame,faceCascade,img_id)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()


except:
    print("An exception occurred")


# faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#
#
# def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
#     coords = []
#     for (x, y, w, h) in features:
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         # coords = [x, y, w, h]
#
#     return img
#
# def detect(img, faceCascade):
#         img, coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 0, 255), "Face")
#         return img
#
# cap = cv2.VideoCapture("This.mp4")
# while (True) :
#     ret , frame = cap.read()
#     # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     frame = detect(frame,faceCascade)
#     cv2.imshow('frame',frame)
#     if(cv2.waitKey(1) & 0xFF == ord('q')):
#         break
# cap.release()
# cv2.destroyAllWindows()
