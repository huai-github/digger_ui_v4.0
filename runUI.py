import sys
from time import sleep
import UI
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from my_thread import MyThread
import multi_thread
import threading
import globalvar as gl

# h = 480  	# 画布大小
# w = 550
from tools import work_area

h = 560  # 画布大小
w = 620

g_startX = 0
g_startY = 0
g_startH = 0
g_startW = 0
g_endX = 0
g_endY = 0
g_endH = 0
g_endW = 0


def get_global_value():
	global g_startX, g_startY, g_startH, g_startW, g_endX, g_endY, g_endH, g_endW
	g_startX = gl.get_value('g_startX')
	g_startY = gl.get_value('g_startY')
	g_startH = gl.get_value('g_startH')
	g_startW = gl.get_value('g_startW')
	g_endX = gl.get_value('g_endX')
	g_endY = gl.get_value('g_endY')
	g_endH = gl.get_value('g_endH')
	g_endW = gl.get_value('g_endW')
	print("g_startX:", g_startX)


class UIFreshThread(object):  # 界面刷新线程
	def __init__(self):
		self.startX = 0  # from could
		self.startY = 0
		self.endX = 0
		self.endY = 0
		self.Interval = 0

		self.nowX = 0  # from gps
		self.nowY = 0
		self.deep = 0

	def __call__(self):  # 调用实例本身 ——>> MyThread(self.__thread,....
		if g_startX != 0:
			self.startX = g_startX  # from could
			self.startY = g_startY
			self.endX = g_endX
			self.endY = g_endY
			self.Interval = g_startW

		self.nowX = multi_thread.g_x - 4076000  # from gps
		self.nowY = multi_thread.g_y - 515000
		self.deep = multi_thread.g_h  # - 基准高 baseHeight from could
		sleep(1)

	def get_msg_xy(self):
		return self.startX, self.startY, self.endX, self.endY, self.Interval, self.nowX, self.nowY

	def get_msg_deep(self):
		return self.deep

	def get_msg_startXY(self):
		return self.startX, self.startY

	def get_msg_endXY(self):
		return self.endX, self.endY

	def get_msg_nowXY(self):
		return self.nowX, self.nowY


class MyWindows(QWidget, UI.Ui_Form):
	def __init__(self):
		super().__init__()
		# 注意：里面的控件对象也成为窗口对象的属性了
		self.setupUi(self)
		self.imgLine = np.zeros((h, w, 3), np.uint8)  # 画布
		self.imgBar = np.zeros((h, w, 3), np.uint8)
		self.figure = plt.figure()  # 可选参数,face color为背景颜色
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

	def leftWindow(self, img, startX, startY, endX, endY, Interval, nowX, nowY):
		img[...] = 255
		startPoint = (startX, startY)
		endPoint = (endX, endY)
		width = Interval * 3
		currentPoint = (nowX, nowY)
		corner = work_area(img, startPoint, endPoint, width, currentPoint)  # 返回矩形的角点坐标，从右上角开始逆时针选择
		# print("corner:", corner)
		border1 = corner[0][0][0]   # 右上角的x坐标
		border2 = corner[1][0][0]
		area1 = corner[0][0][1]     # 右上角的y坐标
		area2 = corner[2][0][1]
		ptx = corner[4][0][0]
		pty = corner[4][0][1]

		BorderReminderLedXY = (w - 25, h - 18)  # 边界指示灯位置 界内绿色
		BorderReminderTextXY = (w - 320, h - 10)
		cv2.circle(img, BorderReminderLedXY, 12, (0, 255, 0), -1)
		self.BorderReminder.setText("   ")

		"""边界和区域判断"""
		# 如果超出边界，BorderReminder红色,并提示汉字信息
		if (ptx < border2) or (ptx > border1):
			cv2.circle(img, BorderReminderLedXY, 12, (0, 0, 255), -1)  # 边界报警指示灯
			self.BorderReminder.setText("！！即将超出边界！！")
		# 如果超出工作区域
		if (pty < area1) or (pty > area2):
			area_inc_flag = True
			gl.set_value("area_inc_flag", area_inc_flag)

			work_area_num = gl.get_value("work_area_num")
			work_area_num += 1
			gl.set_value("work_area_num", work_area_num)

		cv2.putText(img, "BorderReminder", BorderReminderTextXY, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
		QtImgLine = QImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).data,
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

	def showStartXY(self, startX, startY):
		self.startXY.setText("(%.2f, %.2f)" % (startX, startY))

	def showEndXY(self, endX, endY):
		self.endXY.setText("(%.2f, %.2f)" % (endX, endY))

	def showNowXY(self, nowX, nowY):
		self.nowXY.setText("(%.2f, %.2f)" % (nowX, nowY))

	def update(self):
		self.rightWindow(self.imgBar, self.__thread.get_msg_deep())
		self.leftWindow(self.imgLine, *self.__thread.get_msg_xy())
		self.showStartXY(*self.__thread.get_msg_startXY())
		self.showEndXY(*self.__thread.get_msg_endXY())
		self.showNowXY(*self.__thread.get_msg_nowXY())


if __name__ == "__main__":
	gl._init()
	app = QApplication(sys.argv)

	gps_thread = threading.Thread(target=multi_thread.thread_gps_func, daemon=True)
	_4g_thread = threading.Thread(target=multi_thread.thread_4g_func, daemon=True)

	gps_thread.start()  # 启动线程
	_4g_thread.start()
	# sleep(1)

	while True:
		reced_flag = gl.get_value("reced_flag")
		if reced_flag:
			reced_flag = False
			gl.set_value("reced_flag", reced_flag)
			mainWindow = MyWindows()
			get_global_value()
			mainWindow.show()
			sys.exit(app.exec_())
