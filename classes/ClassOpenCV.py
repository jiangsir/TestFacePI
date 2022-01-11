import os
import time
from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2, dlib
import numpy as np
from imutils import face_utils
import imutils

ttf = "C:/Windows.old/Windows/Fonts/msjhbd.ttc"  # 字體: 微軟正黑體

def show_arrow(cv2, shape, img):
    '''
    顯示面部方向箭頭
    '''
    image_points = np.array([
                tuple(shape[30]),#鼻頭
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ],dtype='double')

    
    #cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
    
    model_points = np.array([
            (0.0,0.0,0.0), # 30
            (-30.0,-125.0,-30.0), # 21
            (30.0,-125.0,-30.0), # 22
            (-60.0,-70.0,-60.0), # 39
            (60.0,-70.0,-60.0), # 42
            (-40.0,40.0,-50.0), # 31
            (40.0,40.0,-50.0), # 35
            (-70.0,130.0,-100.0), # 48
            (70.0,130.0,-100.0), # 54
            (0.0,158.0,-10.0), # 57
            (0.0,250.0,-50.0) # 8
            ])

    #size = frame.shape
    size = img.shape

    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2) #顔の中心座標

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')

    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #回転行列とヤコビアン
    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    mat = np.hstack((rotation_matrix, translation_vector))

    #yaw,pitch,rollの取り出し
    (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
    yaw = eulerAngles[1]
    pitch = eulerAngles[0]
    roll = eulerAngles[2]

    print("yaw",int(yaw),"pitch",int(pitch),"roll",int(roll))#頭部姿勢データの取り出し

    cv2.putText(img, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(img, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.putText(img, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                        translation_vector, camera_matrix, dist_coeffs)
    #計算に使用した点のプロット/顔方向のベクトルの表示
    for p in image_points:
        cv2.drawMarker(img, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2)

def show_68points(cv2, shape, img):
    '''
    顯示 68 個特徵點
    '''
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

def show_opencv(hint='', mirror=True):
    ''' 顯示主畫面 '''

    #cam = cv2.VideoCapture(config['videoid'])
    print('cam opening...')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    print('cam opened')
    cam.set(3, 1280)  # 修改解析度 寬
    cam.set(4, 1280 // 16 * 10)  # 修改解析度 高
    print('WIDTH', cam.get(3), 'HEIGHT', cam.get(4))  # 顯示預設的解析度
    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        # image = imutils.resize(img, width=500)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        face_rects = detector(img, 0)

###############################################
        for (i, rect) in enumerate(face_rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            show_68points(cv2, shape, img)
            show_arrow(cv2, shape, img)

        cv2.imshow("Output", img)
###############################################
        # # 取出所有偵測的結果
        # for i, d in enumerate(face_rects):
        #     x1 = d.left()
        #     y1 = d.top()
        #     x2 = d.right()
        #     y2 = d.bottom()

        #     # 以方框標示偵測的人臉
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        H, W = img.shape[:2]

        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 
        pil_im = Image.fromarray(cv2_im)
        draw = ImageDraw.Draw(pil_im)  #

        ##font = ImageFont.truetype(ttf, 40, encoding="utf-8")
        hintfont = ImageFont.truetype(ttf, 24, encoding="utf-8")

        hints = "請按「ENTER」繼續" + hint
        w, h = draw.textsize(hints, font=hintfont)
        draw.rectangle(
            ((W / 2 - w / 2 - 5, H - h), (W / 2 + w / 2 + 5, H)), fill="red")
        hintlocation = (W / 2 - w / 2, H - h)
        #textlocation = (0,0)
        draw.text(
            hintlocation, hints, (0, 255, 255),
            font=hintfont)  #

        cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        # if ClassUtils.isWindows():
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        cv2.imshow("window", cv2_text_im)
        #cv2.imshow("window", img)

        key = cv2.waitKey(1)
        if key == ord(' ') or key == 3 or key == 13:  # space or enter
            # picturepath = getTakePicturePath(
            #     config['personGroupId'])
            # ret_val, img = cam.read()
            # cv2.imwrite(picturepath, img)
            # cv2.destroyAllWindows()
            # cv2.VideoCapture(0).release()
            # return picturepath
            pass
        elif key == 27:  # esc to quit
            cv2.destroyAllWindows()
            cv2.VideoCapture(0).release()
            raise print("偵測到 esc 結束鏡頭")
        else:
            if key != -1:
                print('key=', key)


def show_ImageText(title, hint, facepath=None, picture=None, identifyfaces=None, personname=None):
    ''' 標準 cv 視窗'''
    import cv2
    import numpy as np
    if facepath == None:
        img = np.zeros((400, 400, 3), np.uint8)
        img.fill(90)
    else:
        img = cv2.imread(facepath)
        print('__cv_ImageText.imagepath=', facepath)
        H, W = img.shape[:2]
        img = cv2.resize(img, (400, int(H / W * 400)))

    windowname = facepath
    H, W = img.shape[:2]

    #img = cv2.resize(img, (400,int(H/W*400)))

    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
    titlefont = ImageFont.truetype(ttf, 24, encoding="utf-8")
    hintfont = ImageFont.truetype(ttf, 18, encoding="utf-8")

    w, h = draw.textsize(title, font=titlefont)
    draw.rectangle(
        ((W / 2 - w / 2 - 5, 0), (W / 2 + w / 2 + 5, h + 20)), fill="black")
    titlelocation = (W / 2 - w / 2, 5)

    if identifyfaces != None and len(identifyfaces) == 1:
        hint = hint + "或按 'a' 新增身份"
    w, h = draw.textsize(hint, font=hintfont)
    draw.rectangle(
        ((W / 2 - w / 2 - 5, H - h), (W / 2 + w / 2 + 5, H)), fill="red")
    hintlocation = (W / 2 - w / 2, H - h)
    draw.text(titlelocation, title, (0, 255, 255), font=titlefont)
    draw.text(hintlocation, hint, (0, 255, 0), font=hintfont)

    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imshow(windowname, cv2_text_im)
    key = cv2.waitKey(10000)
    if key == ord(' ') or key == 3 or key == 13:  # space or enter
        cv2.destroyWindow(windowname)
    elif key == ord('a') and len(identifyfaces) == 1:  # 鍵盤 a 代表要新增 oneshot
        cv2.destroyWindow(windowname)
        #ClassTK.tk_UnknownPerson('您哪位？', facepath, picture, personname)

