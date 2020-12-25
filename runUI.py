import sys
import UI
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from time import sleep

import multi_thread
from my_thread import MyThread
from multi_thread import *
import globalvar as gl

# h = 480  	# 画布大小
# w = 550

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
	print("g_startW", g_startW)


class UIFreshThread(object):  # 界面刷新线程
	def __init__(self):
		self.nowX = 0  # from gps
		self.nowY = 0
		self.deep = 0

	def __call__(self):  # 调用实例本身 ——>> MyThread(self.__thread,....
		self.nowX = multi_thread.g_x - 4076000  # from gps
		self.nowY = multi_thread.g_y - 515000
		self.deep = multi_thread.g_h  # - 基准高 baseHeight from could

	def get_msg_deep(self):
		return self.deep

	def get_msg_nowXY(self):
		return self.nowX, self.nowY


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

	def leftWindow(self, img, sx_list, sy_list, ex_list, ey_list, width, nowX, nowY):
		img[...] = 255  # 画布
		interval = width  # 宽度
		currentPoint = (nowX, nowY)

		x_min = min(sx_list)
		y_min = min(sy_list)

		for i in range(len(sx_list)):
			sx = sx_list[i] - x_min + 50
			sy = sy_list[i] - y_min + 50
			ex = ex_list[i] - x_min + 50
			ey = ey_list[i] - y_min + 50
			start_point = (int(sx), int(sy))  # 画中线
			end_point = (int(ex), int(ey))
			cv.line(img, start_point, end_point, (0, 255, 0), 2)

			k = (ey - sy) / (ex - sx)
			theta = np.arctan(k)
			x_offset = interval * sin(theta) * -1
			y_offset = interval * cos(theta)
			M = np.float32([[1, 0, x_offset],
			                [0, 1, y_offset]])
			new_pt = cv.transform(np.float32([[start_point], [end_point]]), M).astype(np.int)

			x1_offset = interval * sin(theta)
			y1_offset = interval * cos(theta) * -1
			M1 = np.float32([[1, 0, x1_offset],
			                 [0, 1, y1_offset]])
			new_pt1 = cv.transform(np.float32([[start_point], [end_point]]), M1).astype(np.int)

			box = np.array([tuple(new_pt1[0][0]), tuple(new_pt[0][0]), tuple(new_pt[1][0]), tuple(new_pt1[1][0])])
			cv.drawContours(img, [box[:, np.newaxis, :]], 0, (0, 0, 255), 2)

			cv.circle(img, currentPoint, 5, [0, 0, 255], -1)
			dist = cv.pointPolygonTest(box, currentPoint, False)
			print("dist", dist)

		BorderReminderLedXY = (w - 25, h - 18)  # 边界指示灯位置 界内绿色
		BorderReminderTextXY = (w - 320, h - 10)
		cv2.circle(img, BorderReminderLedXY, 12, (0, 255, 0), -1)
		self.BorderReminder.setText("   ")

		# TODO：如果超出边界，BorderReminder红色,并提示汉字信息

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

	def showNowXY(self, nowX, nowY):
		self.nowXY.setText("(%.2f, %.2f)" % (nowX, nowY))

	def update(self):
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
		# TODO: 宽度怎么办？？

		current_x, current_y = self.__thread.get_msg_nowXY()
		self.leftWindow(self.imgLine, g_start_x_list, g_start_y_list, g_end_x_list, g_end_y_list, g_start_w_list[0],
		                int(current_x),
		                int(current_y))
		# TODO：如果绘制全部的直线段，显示起点坐标和终点坐标是没有意义
		self.showNowXY(current_x, current_y)


if __name__ == "__main__":
	gl.gl_init()
	app = QApplication(sys.argv)

	gps_thread = threading.Thread(target=multi_thread.thread_gps_func, daemon=True)
	_4g_thread = threading.Thread(target=multi_thread.thread_4g_func, daemon=True)
	# gyro_thread = threading.Thread(target=multi_thread.thread_gyro_func, daemon=True)
	# g_laser1_thread = threading.Thread(target=multi_thread.thread_laser1_func, daemon=True)
	# g_laser2_thread = threading.Thread(target=multi_thread.thread_laser2_func, daemon=True)
	# g_laser3_thread = threading.Thread(target=multi_thread.thread_laser3_func, daemon=True)
	# calculate_thread = threading.Thread(target=calculate.altitude_calculate_func, daemon=True)

	gps_thread.start()  # 启动线程
	mainWindow = MyWindows()
	_4g_thread.start()
	# gyro_thread.start()
	# calculate_thread.start()

	while True:
		reced_flag = gl.get_value("reced_flag")
		if reced_flag:
			reced_flag = False
			gl.set_value("reced_flag", reced_flag)
			mainWindow = MyWindows()
			get_global_value()
			# sleep(1)
			mainWindow.show()
			sys.exit(app.exec_())
