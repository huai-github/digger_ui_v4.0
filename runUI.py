import sys
from math import sqrt

import UI
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from time import sleep

import calculate
import multi_thread
from my_thread import MyThread
from multi_thread import *
import globalvar as gl

g_ui_threadLock = threading.Lock()

h = 480  # 画布大小
w = 550

# h = 560  # 画布大小
# w = 620

x_min = 0
y_min = 0


def y1(d, x1, x0, y0):
    return y0 + sqrt((d ** 2) + (x1 - x0) ** 2)


class UIFreshThread(object):  # 界面刷新线程
    def __init__(self):
        self.nowX = 0  # from gps
        self.nowY = 0
        self.deep = 0

    def __call__(self):  # 调用实例本身 ——>> MyThread(self.__thread,....
        self.nowX = multi_thread.g_x  # from gps
        self.nowY = multi_thread.g_y
        self.deep = gl.get_value("h0") - multi_thread.g_h

    def get_msg_deep(self):
        return self.deep

    def get_msg_nowXY(self):
        return self.nowX, self.nowY


def show_img(name, img, delay=0):
    cv.imshow(name, img)
    if cv.waitKey(delay) == 27:
        cv.destroyAllWindows()
        exit(0)


class MyWindows(QWidget, UI.Ui_Form):
    def __init__(self):
        super().__init__()
        # 注意：里面的控件对象也成为窗口对象的属性了
        self.setupUi(self)
        self.imgLine = np.zeros((h, w, 3), np.uint8)  # 画布
        self.imgBar = np.zeros((h, w, 3), np.uint8)
        self.figure = plt.figure()  # 可选参数,facecolor为背景颜色
        self.canvas = FigureCanvas(self.figure)
        self.__timer = QtCore.QTimer()  # 定时器用于定时刷新
        self.set_slot()
        self.__thread = UIFreshThread()  # 开启线程(同时将这个线程类作为一个属性)
        MyThread(self.__thread, (), name='UIFreshThread', daemon=True).start()
        self.__timer.start(1000)  # ms
        self.DeepList = []
        self.NumList = []

    def set_slot(self):
        self.__timer.timeout.connect(self.update)

    def leftWindow(self, img, sx_list, sy_list, ex_list, ey_list, s_width, e_width, nowX, nowY):
        line_shift = 7  # 画直线时小数点的位数
        img[...] = 255  # 画布
        below_start = 0
        below_end = 0
        above_start = 0
        above_end = 0

        currentPoint = (nowX, nowY)
        global x_min, y_min
        x_min = min(sx_list)
        y_min = min(sy_list)

        last_lines = [None, None]  # 保存上一条直线
        for i in range(len(sx_list)):
            sx = sx_list[i] - x_min + 50  # sx = x0
            sy = sy_list[i] - y_min + 50
            ex = ex_list[i] - x_min + 50
            ey = ey_list[i] - y_min + 50

            start_point = (int(sx), int(sy))  # 画中线
            end_point = (int(ex), int(ey))

            cv.line(img, start_point, end_point, (0, 255, 0), 2)  # 画中线

            k = (ey - sy) / (ex - sx)
            theta = np.arctan(k)

            """计算点的坐标"""
            # 下侧直线
            x_below_s = s_width[i] * sin(theta) * -1  # 根据中线偏移的直线坐标
            y_below_s = s_width[i] * cos(theta)
            M = np.float32([[1, 0, x_below_s],
                            [0, 1, y_below_s]])
            start_below_pt = cv.transform(np.float32([[start_point], [end_point]]), M).astype(np.int)
            # 起点：start_below_pt[0][0]， 终点：start_below_pt[1][0]
            # below_start = tuple(start_below_pt[0][0])  # 只要起点坐标，舍掉终点，重新计算终点坐标
            below_start = start_below_pt[0][0]  # 只要起点坐标，舍掉终点，重新计算终点坐标

            x_below_e = e_width[i] * sin(theta) * -1  # 根据中线偏移的直线坐标
            y_below_e = e_width[i] * cos(theta)
            M = np.float32([[1, 0, x_below_e],
                            [0, 1, y_below_e]])
            end_below_pt = cv.transform(np.float32([[start_point], [end_point]]), M).astype(np.int)
            # below_end = tuple(end_below_pt[1][0])
            below_end = end_below_pt[1][0]

            # 上侧直线
            x_above_s = s_width[i] * sin(theta)  # 根据中线偏移的直线坐标
            y_above_s = s_width[i] * cos(theta) * -1
            M = np.float32([[1, 0, x_above_s],
                            [0, 1, y_above_s]])
            start_above_pt = cv.transform(np.float32([[start_point], [end_point]]), M).astype(np.int)
            # above_start = tuple(start_above_pt[0][0])
            above_start = start_above_pt[0][0]

            x_above_e = e_width[i] * sin(theta)  # 根据中线偏移的直线坐标
            y_above_e = e_width[i] * cos(theta) * -1
            M = np.float32([[1, 0, x_above_e],
                            [0, 1, y_above_e]])
            end_above_pt = cv.transform(np.float32([[start_point], [end_point]]), M).astype(np.int)
            # above_end = tuple(end_above_pt[1][0])
            above_end = end_above_pt[1][0]

            # cv.line(img, tuple(below_start), tuple(below_end), (0, 0, 255), 1)
            # cv.line(img, tuple(above_start), tuple(above_end), (0, 0, 255), 1)
            pts = np.array([
                [above_start],
                [above_end],
                [below_end],
                [below_start],
            ], np.int32)

            # 计算当前线段的斜率
            below_k = (below_end[1] - below_start[1]) / (below_end[0] - below_start[0])
            below_b = below_start[1] - (below_k * below_start[0])
            above_k = (above_end[1] - above_start[1]) / (above_end[0] - above_start[0])
            above_b = above_start[1] - (above_k * above_start[0])

            if i > 0:
                # 获得上一条线的斜率和后端点
                before_above_pt, before_above_k, before_above_b = last_lines[0]
                before_below_pt, before_below_k, before_below_b = last_lines[1]

                # 断开前一步闭合的连线
                cv.line(img, before_above_pt, before_below_pt, (255, 255, 255), 2)
                # show_img("delete line", img)

                # 计算交点
                intersection_above_pt_x = (above_b - before_above_b) / (before_above_k - above_k)
                intersection_above_pt_y = above_k * intersection_above_pt_x + above_b
                intersection_below_pt_x = (below_b - before_below_b) / (before_below_k - below_k)
                intersection_below_pt_y = below_k * intersection_below_pt_x + below_b

                intersection_above_pt = (int(intersection_above_pt_x), int(intersection_above_pt_y))
                intersection_below_pt = (int(intersection_below_pt_x), int(intersection_below_pt_y))
                # cv.circle(img, intersection_above_pt, 3, (255, 0, 0), -1)
                # cv.circle(img, intersection_below_pt, 3, (255, 0, 0), -1)
                # show_img("circle", img)

                # 连接上边界的终点与交点
                cv.line(img, before_above_pt, intersection_above_pt, (255, 0, 255), 2)
                # show_img("up line", img)
                # 删除下边界的终点与交点
                cv.line(img, before_below_pt, intersection_below_pt, (255, 255, 255), 2)
                # show_img("down line", img)

                pts[0] = intersection_above_pt  # 将上线段的起点替换为交点
                pts[-1] = intersection_below_pt  # 将下线段的起点替换为交点

            is_closed = False if i > 0 else True
            cv.polylines(img, [pts], is_closed, (255, 0, 255), 2)  # 闭合
            # show_img("img", img)

            # 保存本边界的斜率和终点
            last_lines[0] = (tuple(above_end), above_k, above_b)
            last_lines[1] = (tuple(below_end), below_k, below_b)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)  # 转为二值图
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 画出轮廓线
        # draw_img = img.copy()
        # res = cv.drawContours(draw_img, contours, 1, (0, 0, 255), 2)
        # cv.imshow('res', res)

        box = contours[1]

        currentPoint = (70, 50)
        cv.circle(img, currentPoint, 5, [0, 0, 255], -1)

        # It returns positive (inside), negative (outside), or zero (on an edge)
        dist = cv.pointPolygonTest(box, currentPoint, False)
        # print("dist:", dist)

        BorderReminderLedXY = (w - 25, h - 18)  # 边界指示灯
        BorderReminderTextXY = (w - 320, h - 10)
        cv.circle(img, BorderReminderLedXY, 12, (0, 255, 0), -1)
        cv.putText(img, "BorderReminder", BorderReminderTextXY, cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        self.BorderReminder.setText(" ")

        if dist == -1:
            cv.circle(img, BorderReminderLedXY, 12, (0, 0, 255), -1)  # 边界报警指示灯
            self.BorderReminder.setText("！！即将超出边界！！")

        QtImgLine = QImage(cv.cvtColor(img, cv.COLOR_BGR2RGB).data,
                           img.shape[1],
                           img.shape[0],
                           img.shape[1] * 3,  # 每行的字节数, 彩图*3
                           QImage.Format_RGB888)

        pixmapL = QPixmap(QtImgLine)
        self.leftLabel.setPixmap(pixmapL)

    def rightWindow(self, img, deep):
        img[::] = 255  # 设置画布颜色

        if len(self.NumList) >= 5:  # 最多显示5条柱状图
            self.DeepList.pop(0)
            self.NumList.pop(0)

        self.DeepList.append(deep)
        self.NumList.append(' ')

        # 将self.DeepList中的数据转化为int类型
        self.DeepList = list(map(float, self.DeepList))

        # 将x,y轴转化为矩阵式
        self.x = np.arange(len(self.NumList)) + 1
        self.y = np.array(self.DeepList)

        colors = ["g" if i > 0 else "r" for i in self.DeepList]
        plt.clf()
        plt.bar(range(len(self.NumList)), self.DeepList, tick_label=self.NumList, color=colors, width=0.5)

        # 在柱体上显示数据
        for a, b in zip(self.x, self.y):
            plt.text(a - 1, b, '%.4f' % b, ha='center', va='bottom')

        # 画图
        self.canvas.draw()

        img = np.array(self.canvas.renderer.buffer_rgba())

        QtImgBar = QImage(img.data,
                          img.shape[1],
                          img.shape[0],
                          img.shape[1] * 4,
                          QImage.Format_RGBA8888)
        pixmapR = QPixmap(QtImgBar)

        self.rightLabel.setPixmap(pixmapR)

    def showNowXY(self, nowX, nowY):
        self.nowXY.setText("(%.2f, %.2f)" % (nowX, nowY))

    def update(self):
        g_ui_threadLock.acquire()
        global x_min, y_min
        worked_flag = gl.get_value("worked_flag")
        if worked_flag:
            self.rightWindow(self.imgBar, self.__thread.get_msg_deep())
        g_start_x_list = gl.get_value('g_start_x_list')  # [122.22, 32.33]
        g_start_y_list = gl.get_value('g_start_y_list')
        g_start_h_list = gl.get_value('g_start_h_list')
        g_start_w_list = gl.get_value('g_start_w_list')
        g_end_x_list = gl.get_value('g_end_x_list')
        g_end_y_list = gl.get_value('g_end_y_list')
        g_end_h_list = gl.get_value('g_end_h_list')
        g_end_w_list = gl.get_value('g_end_w_list')
        # print('g_end_w_list:', g_end_w_list)

        current_x, current_y = self.__thread.get_msg_nowXY()
        self.leftWindow(self.imgLine, g_start_x_list, g_start_y_list, g_end_x_list, g_end_y_list,
                        g_start_w_list,
                        g_end_w_list,
                        int(current_x),
                        int(current_y))

        self.showNowXY(current_x - x_min, current_y - y_min)
        g_ui_threadLock.release()


if __name__ == "__main__":
    gl.gl_init()
    app = QApplication(sys.argv)

    gps_thread = threading.Thread(target=multi_thread.thread_gps_func, daemon=True)
    _4g_thread = threading.Thread(target=multi_thread.thread_4g_func, daemon=True)
    gyro_thread = threading.Thread(target=multi_thread.thread_gyro_func, daemon=True)
    g_laser1_thread = threading.Thread(target=multi_thread.thread_laser1_func, daemon=True)
    # g_laser2_thread = threading.Thread(target=multi_thread.thread_laser2_func, daemon=True)
    # g_laser3_thread = threading.Thread(target=multi_thread.thread_laser3_func, daemon=True)
    calculate_thread = threading.Thread(target=calculate.altitude_calculate_func, daemon=True)

    # gps_thread.start()  # 启动线程
    mainWindow = MyWindows()
    _4g_thread.start()
    gyro_thread.start()
    g_laser1_thread.start()
    calculate_thread.start()

    while True:
        reced_flag = gl.get_value("reced_flag")
        if reced_flag:
            reced_flag = False
            gl.set_value("reced_flag", reced_flag)
            mainWindow = MyWindows()
            mainWindow.show()
            sys.exit(app.exec_())
