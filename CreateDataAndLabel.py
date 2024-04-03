import os
import cv2
import mediapipe as mp
from time import time

# ----------
classID = 1# 0: FAKE, 1: REAL
confidence = 0.8
save = True
debug = False
outputFolderPath = 'Dataset/Real'
# outputFolderPath = 'Dataset/Fake'
blurThreshold = 30
# ----------
offsetPercentageW = 10
offsetPercentageH = 20
floatingPoint = 6

# ---------- Đặt kích thước mong muốn cho camera ----------
# desired_width = 288 
# desired_height = 352

# ---------- PATH ----------
os.makedirs(outputFolderPath, exist_ok=True) #exist_ok=True để không báo lỗi nếu thư mục đã tồn tại

cap = cv2.VideoCapture(0)
# ----------Thiết lập kích thước cho khung hình camera----------
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_save = img.copy()
    listBlur = []
    listInfo = []
    
    
    with mp_face_detection.FaceDetection() as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                score = detection.score[0]
                
                # ---------- Check score ----------
                if score > confidence:
                    # ---------- Mở rộng viền box ----------
                    offsetW = (offsetPercentageW / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)

                    offsetH = (offsetPercentageH / 100) * h
                    y = int(y - offsetH * 2.5)
                    h = int(h + offsetH * 3)

                    # ---------- Tránh lỗi khi các giá trị dưới 0 ----------
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if w < 0: w = 0
                    if h < 0: h = 0

                    # ---------- Tìm điểm mờ (Blur) của ảnh ----------
                    imgFace = img[y:y + h, x:x + w]
                    cv2.imshow("imgface", imgFace)
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    print(blurValue)
                    if blurValue > blurThreshold:
                        listBlur.append(True)
                    else:
                        listBlur.append(False)

                    # ---------- Chuẩn hoá (Normalize Values) ----------
                    xc, yc = x + w / 2, y + h / 2
                    xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                    wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                    print(xcn, ycn, wn, hn)

                    # ---------- Tránh lỗi khi các giá trị trên 1 ----------
                    if xcn > 1: xcn = 1
                    if ycn > 1: ycn = 1
                    if wn > 1: wn = 1
                    if hn > 1: hn = 1

                    # ---------- Thêm các giá trị Nomarlize từng dòng với classID tại [0] là label ----------
                    listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn}\n')

                    # ---------- Drawing ----------
                    cv2.rectangle(img_save, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(img_save, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # ---------- Hiển thị debug ảnh ----------
                    if debug:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # ---------- Save ----------
            if save:
                if all(listBlur) and listBlur != []:
                # ---------- Save img với khoảng thời gian hiện tại ----------
                    timeNow = time()
                    timeNow = str(timeNow).split('.')
                    timeNow = timeNow[0] + timeNow[1]
                    cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg', img)

                    # ---------- Save labels .txt file ----------
                    for info in listInfo:
                        f = open(f'{outputFolderPath}/{timeNow}.txt', 'a')
                        f.write(info)
                        f.close()

    cv2.imshow("img", img_save)
    if cv2.waitKey(1) & 0xFF == 27: #Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
