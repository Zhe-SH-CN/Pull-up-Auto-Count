import cv2
import mediapipe as mp

# 这个库是为了计算fps的
import time

mp_drawing = mp.solutions.drawing_utils  # drawing_utils是画图工具 目的是为了能画出点
mp_drawing_styles = mp.solutions.drawing_styles  # 画的样式
mp_pose = mp.solutions.pose  # 表示我们要的是pose 即姿势识别

# 为了计算fps
pTime = 0  # previous time 之前的时间
cTime = 0  # current time 目前的时间

# 引体向上计数
PullUpCount = 0
# 记录当前先前头是不是在手的上面，不记录的话可能上去的时候无限制符合
isUp = False
# 最开始的时候，头会在上面，这时候不要计入
FirstUp = True

# For webcam input: 捕捉摄像头的或者视频文件里的
cap = cv2.VideoCapture(0)  # 用opencv调用摄像头
with mp_pose.Pose(
        min_detection_confidence=0.5,  # 最低侦测严谨度 值越大，侦测的时候越严谨（判定是不是人体的要求更严格，值太大会出现是人却识别不出，值太小会出现不是人但是识别出人）（值在[0,1)之间)
        min_tracking_confidence=0.5  # 最低追踪严谨度 值越大，追踪的时候越严谨（越有可能要重新追踪，值太大性能耗费太大，值变小的话会导致人体移动了但是没有更新位置）（值在[0,1)之间)
) as pose:
    while cap.isOpened():  # 判断摄像头是不是开着
        success, image = cap.read()  # success是是否成功开启的状态，image是摄像头捕捉到的图片
        if not success:  # 如果不成功
            print("摄像头被占用")  # 摄像头被占用
            # 如果加载的是视频，就用break 而不是continue
            continue

        # 为了提高性能，可选择将图像标记为不可写入
        # 通过引用传递 就是这俩是一份 没有复制出来新的一份
        image.flags.writeable = False
        # 把bgr转为rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        # 变为可写入
        image.flags.writeable = True
        # 转bgr为rgb
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 如果捕捉到了关键点
        # if results.pose_landmarks:
        # mp_drawing.draw_landmarks(  # 画点
        #     image,  # 背景图
        #     results.pose_landmarks,  # 每个点
        #     mp_pose.POSE_CONNECTIONS,  # 点之间的线
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        # )  # 这里是给了画点的样式 就是default 默认。前面landmark_drawing_spec表示我们要传给的是这个参数
        #
        # # 定位每个关键点 id即为编号，lm即landmark 关键点
        # for plotId, lm in enumerate(results.pose_landmarks.landmark):
        #     # 获取视频高 宽 通道数
        #     height, width, channel = image.shape
        #     # 计算关键点的x,y坐标
        #     plotX, plotY = int(lm.x * width), int(lm.y * height)
        #     # print(plotId, plotX, plotY)
        #
        #     # 对某一个特定的坐标进行操作
        #     # 第一个坐标是图片，第二个是中心，第三个是半径，第四个是颜色，第五个是粗细 第六个是线的样式
        #     # cv2.circle(image, (plotX, plotY), 5, (255, 0, 0), cv2.FILLED)
        #
        #     # 给每个关键点显示编号
        #     cv2.putText(image, str(int(plotId)), (plotX - 20, plotY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # 由定位得到
        # 左手掌心关键点编号 17 19
        # 右手掌心关键点编号 18 20
        # 嘴巴关键点编号 9 10
        LeftHand = results.pose_landmarks.landmark[19]
        RightHand = results.pose_landmarks.landmark[20]
        MouthLeft = results.pose_landmarks.landmark[9]
        MouthRight = results.pose_landmarks.landmark[10]
        # 视频的长宽 通道数
        height, width, channel = image.shape

        # 手的高度
        HandHeight = max(LeftHand.y * height, RightHand.y * height)
        # 嘴的高度
        MouthHeight = min(MouthLeft.y * height, MouthRight.y * height)

        # 当头超过手的时候 注意坐标越向下越大
        if HandHeight > MouthHeight:
            # 当之前在下面的时候
            if not isUp and not FirstUp:
                isUp = True
                PullUpCount += 1
                # print(PullUpCount)
        else:
            if FirstUp:
                FirstUp = False
            # 要是原本头在上面，说明这时候下来了
            if isUp:
                isUp = False
            # 头原本不在上面，那么就没什么关系

        # 显示个数 中文有乱码
        cv2.putText(image, f"Pull-Up Count:{PullUpCount}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # 计算fps
        cTime = time.time()  # 获取当前时间
        fps = 1 / (cTime - pTime)  # fps是每秒刷新几次 就是 1秒/每次刷新的时间 cTime-pTime就是这次刷新完时间-上次刷新完时间，就是每次刷新的时间
        pTime = cTime  # 更新一下时间

        # 显示fps
        # putText是写文本的函数，第一个参数是要写在什么图片上 第二个参数是写什么内容，第三个参数是写的位置，这里是(30,50) 横坐标为30，纵坐标50的地方写，
        # 第四个参数是字体，第五个参数是字的大小，第六个参数是字的颜色，采用bgr颜色，第七个参数是字的粗细
        cv2.putText(image, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Flip the image horizontally for a selfie-view display.
        #  MediaPipe pose就是显示的标题 也可以 水平翻转图像以供稍后自拍视图显示 把image改为cv2.flip(image,1)
        cv2.imshow('Pull-Up Count by ZHE-SH-CN', image)

        # if cv2.waitKey(5) & 0xFF == ord('q')
        if cv2.waitKey(5) & 0xFF == 27:  # 27是ASCII码中的esc 这里waitKey是等待按下键盘，然后(5)表示等5毫秒，&0xFF 是为了取后八位
            break
cap.release()  # 释放捕捉
