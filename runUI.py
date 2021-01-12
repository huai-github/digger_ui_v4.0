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
x_max = 0
y_max = 0

zoom_x = 0
zoom_y = 0
delta = 5


class UIFreshThread(object):  # 界面刷新线程
	def __init__(self):
		self.nowX = 0  # from gps
		self.nowY = 0
		self.deep = 0

	def __call__(self):  # 调用实例本身 ——>> MyThread(self.__thread,....
		self.nowX = multi_thread.g_x  # from gps
		self.nowY = multi_thread.g_y
		h_o = gl.get_value("h_o")
		# base_h = gl.get_value("base_h")
		g_start_h_list = gl.get_value("g_start_h_list")

		# print("h_o-2", h_o)
		# print("base_h", base_h)
		# print("g_start_h_list", g_start_h_list[0])
		if h_o is not None and g_start_h_list is not None:
			# self.deep = h_o - g_start_h_list[0]
			self.deep = h_o
			# self.deep = h_o - base_h

		# print("deep:%s\n" % self.deep)

	def get_msg_deep(self):
		return self.deep

	def get_msg_nowXY(self):
		return self.nowX, self.nowY


def show_img(name, img, delay=0):
	cv.imshow(name, img)
	if cv.waitKey(delay) == 27:
		cv.destroyAllWindows()
		exit(0)


# 判断特殊矩形：矩形的边平行于坐标轴
def isInParRect(x1, y1, x4, y4, x, y):
	# （x1，y1）为最左的点、（x2，y2）为最上的点、（x3，y3）为最下的点、（x4， y4）为最右的点
	# 按顺时针点的位置依次为1，2，4，3
	if x <= x1:
		return False
	if x >= x4:
		return False
	if y >= y1:
		return False
	if y <= y4:
		return False
	return True


def isInRect(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
	# 使一般矩形旋转，使之平行于坐标轴
	# （x1，y1）为最左的点、（x2，y2）为最上的点、（x3，y3）为最下的点、（x4， y4）为最右的点
	# 按顺时针点的位置依次为1，2，4，3

	if x1 != x4:
		# 坐标系以(x3, y3)为中心，逆时针旋转t至(x4, y4)
		dx = x4 - x3
		dy = y4 - y3
		ds = (dx ** 2 + dy ** 2) ** 0.5
		cost = dx / ds
		sint = dy / ds
		# python特性：隐含临时变量存储值
		x, y = cost * x + sint * y, -sint * x + cost * y
		x1, y1 = cost * x1 + sint * y1, -sint * x1 + cost * y1
		x4, y4 = cost * x4 + sint * y4, -sint * x4 + cost * y4
	return isInParRect(x1, y1, x4, y4, x, y)


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
		self.__timer.start(50)  # ms
		self.DeepList = [0, 0, 0, 0, 0]
		self.NumList = [0, 0, 0, 0, 0]

	def set_slot(self):
		self.__timer.timeout.connect(self.update)

	def leftWindow(self, img, sx_list, sy_list, ex_list, ey_list, s_width, e_width, nowX, nowY):
		img[...] = 255  # 画布

		currentPoint = [nowX, nowY]
		currentPoint_move = [None, None]
		currentPoint_zoom = [None, None]

		last_lines = [None, None]  # 保存上一条直线

		sx_list2 = []
		sy_list2 = []
		ex_list2 = []
		ey_list2 = []

		save_line_point_sx_l_list = []
		save_line_point_ex_l_list = []
		save_line_point_sx_r_list = []
		save_line_point_ex_r_list = []
		save_line_point_sy_l_list = []
		save_line_point_ey_l_list = []
		save_line_point_sy_r_list = []
		save_line_point_ey_r_list = []

		save_intersection_xl = []
		save_intersection_xr = []
		save_intersection_yl = []
		save_intersection_yr = []

		"""求出所有点"""
		# 判断k是否存在
		for i in range(len(sx_list)):
			line_point_s = np.array([sx_list[i], sy_list[i]])
			line_point_e = np.array([ex_list[i], ey_list[i]])

			median_k = (line_point_s[1] - line_point_e[1]) / (line_point_s[0] - line_point_e[0])
			median_b = line_point_s[1] - median_k * line_point_s[0]
			# median_G = np.array([-median_k, 1])
			median_G = np.array([-(line_point_e[1] - line_point_s[1]), (line_point_e[0] - line_point_s[0])])

			# 平移出四个点
			line_point_s_l = line_point_s + s_width[i] * median_G / cv.norm(median_G)
			line_point_s_r = line_point_s - s_width[i] * median_G / cv.norm(median_G)
			line_point_e_l = line_point_e + e_width[i] * median_G / cv.norm(median_G)
			line_point_e_r = line_point_e - e_width[i] * median_G / cv.norm(median_G)

			# 求平移出的直线方程
			kl = (line_point_s_l[1] - line_point_e_l[1]) / (line_point_s_l[0] - line_point_e_l[0])
			bl = line_point_s_l[1] - (kl * line_point_s_l[0])
			kr = (line_point_s_r[1] - line_point_e_r[1]) / (line_point_s_r[0] - line_point_e_r[0])
			br = line_point_s_r[1] - (kr * line_point_s_r[0])

			# 两直线交点
			if i > 0:
				before_line_point_e_l, before_kl, before_bl = last_lines[0]
				before_line_point_e_r, before_kr, before_br = last_lines[1]

				xl = (bl - before_bl) / (before_kl - kl)
				yl = kl * xl + bl
				xr = (br - before_br) / (before_kr - kr)
				yr = kr * xr + br

				# 保存交点坐标
				save_intersection_xl.append(xl)
				save_intersection_yl.append(yl)
				save_intersection_xr.append(xr)
				save_intersection_yr.append(yr)

			# 保存本边界的斜率和终点
			last_lines[0] = (tuple(line_point_e_l), kl, bl)
			last_lines[1] = (tuple(line_point_e_r), kr, br)
			# 保存平移出的4个点坐标
			save_line_point_sx_l_list.append(line_point_s_l[0])
			save_line_point_sx_r_list.append(line_point_s_r[0])
			save_line_point_ex_l_list.append(line_point_e_l[0])
			save_line_point_ex_r_list.append(line_point_e_r[0])

			save_line_point_sy_l_list.append(line_point_s_l[1])
			save_line_point_sy_r_list.append(line_point_s_r[1])
			save_line_point_ey_l_list.append(line_point_e_l[1])
			save_line_point_ey_r_list.append(line_point_e_r[1])

		# 所有点 = 中线坐标 + 平移出的坐标 + 交点坐标
		x_list = (sx_list + ex_list) \
		         + (save_line_point_sx_l_list + save_line_point_sx_r_list + save_line_point_ex_l_list + save_line_point_ex_r_list) \
		         + (save_intersection_xl + save_intersection_xr)
		# x_list.append(currentPoint[0])

		y_list = (sy_list + ey_list) \
		         + (save_line_point_sy_l_list + save_line_point_sy_r_list + save_line_point_ey_l_list + save_line_point_ey_r_list) \
		         + (save_intersection_yl + save_intersection_yr)
		# y_list.append(currentPoint[1])

		# 平移所有点
		x_min = min(x_list)
		y_min = min(y_list)
		x_max = max(x_list)
		y_max = max(y_list)

		currentPoint_move[0] = currentPoint[0] - x_min
		currentPoint_move[1] = currentPoint[1] - y_min

		x_list[:] = [v - x_min for v in x_list]
		y_list[:] = [v - y_min for v in y_list]

		sx_list2[:] = [v - x_min for v in sx_list]
		sy_list2[:] = [v - y_min for v in sy_list]

		ex_list2[:] = [v - x_min for v in ex_list]
		ey_list2[:] = [v - y_min for v in ey_list]

		save_line_point_sx_l_list[:] = [v - x_min for v in save_line_point_sx_l_list]
		save_line_point_sy_l_list[:] = [v - y_min for v in save_line_point_sy_l_list]

		save_line_point_sx_r_list[:] = [v - x_min for v in save_line_point_sx_r_list]
		save_line_point_sy_r_list[:] = [v - y_min for v in save_line_point_sy_r_list]

		save_line_point_ex_l_list[:] = [v - x_min for v in save_line_point_ex_l_list]
		save_line_point_ey_l_list[:] = [v - y_min for v in save_line_point_ey_l_list]

		save_line_point_ex_r_list[:] = [v - x_min for v in save_line_point_ex_r_list]
		save_line_point_ey_r_list[:] = [v - y_min for v in save_line_point_ey_r_list]

		save_intersection_xl[:] = [v - x_min for v in save_intersection_xl]
		save_intersection_yl[:] = [v - y_min for v in save_intersection_yl]

		save_intersection_xr[:] = [v - x_min for v in save_intersection_xr]
		save_intersection_yr[:] = [v - y_min for v in save_intersection_yr]

		# 找xy的最大差值
		x_delta_max = x_max - x_min
		y_delta_max = y_max - y_min

		# 缩放因子
		global zoom_x, zoom_y
		zoom_x = ((w - 30) / x_delta_max)
		zoom_y = ((h - 30) / y_delta_max)

		# 所有点乘以系数
		currentPoint_zoom[0] = currentPoint_move[0] * zoom_x + delta
		currentPoint_zoom[1] = currentPoint_move[1] * zoom_y + delta

		x_list[:] = [v * zoom_x + delta for v in x_list]
		y_list[:] = [v * zoom_y + delta for v in y_list]

		sx_list2[:] = [v * zoom_x + delta for v in sx_list2]
		sy_list2[:] = [v * zoom_y + delta for v in sy_list2]

		ex_list2[:] = [v * zoom_x + delta for v in ex_list2]
		ey_list2[:] = [v * zoom_y + delta for v in ey_list2]

		save_line_point_sx_l_list[:] = [v * zoom_x + delta for v in save_line_point_sx_l_list]
		save_line_point_sy_l_list[:] = [v * zoom_y + delta for v in save_line_point_sy_l_list]

		save_line_point_ex_l_list[:] = [v * zoom_x + delta for v in save_line_point_ex_l_list]
		save_line_point_ey_l_list[:] = [v * zoom_y + delta for v in save_line_point_ey_l_list]

		save_line_point_sx_r_list[:] = [v * zoom_x + delta for v in save_line_point_sx_r_list]
		save_line_point_sy_r_list[:] = [v * zoom_y + delta for v in save_line_point_sy_r_list]

		save_line_point_ex_r_list[:] = [v * zoom_x + delta for v in save_line_point_ex_r_list]
		save_line_point_ey_r_list[:] = [v * zoom_y + delta for v in save_line_point_ey_r_list]

		save_intersection_xl[:] = [v * zoom_x + delta for v in save_intersection_xl]
		save_intersection_yl[:] = [v * zoom_y + delta for v in save_intersection_yl]

		save_intersection_xr[:] = [v * zoom_x + delta for v in save_intersection_xr]
		save_intersection_yr[:] = [v * zoom_y + delta for v in save_intersection_yr]

		"""判断点的位置"""
		for i in range(len(sx_list)):
			in_flag = isInRect(
				save_line_point_sx_r_list[i], save_line_point_sy_r_list[i],
				save_line_point_sx_l_list[i], save_line_point_sy_l_list[i],
				save_line_point_ex_r_list[i], save_line_point_ey_r_list[i],
				save_line_point_ex_l_list[i], save_line_point_ey_l_list[i],
				currentPoint_zoom[0], currentPoint_zoom[1]
			)
			if in_flag:
				print("当前工作在第%d段" % (i + 1))
			else:
				print("！！超出工作区域！！")
				pass
		"""画线"""
		# 中线
		for i in range(len(sx_list2)):
			cv.line(img, (int(sx_list2[i]), int(sy_list2[i])), (int(ex_list2[i]), int(ey_list2[i])), (0, 255, 0), 1)

		"""如果交点存在"""
		if save_intersection_xl:
			# 下面偏移的线
			# 地点到第一个交点
			cv.line(img,
			        (int(save_line_point_sx_l_list[0]), int(save_line_point_sy_l_list[0])),
			        (int(save_intersection_xl[0]), int(save_intersection_yl[0])),
			        (0, 0, 255),
			        2)
			# 交点之间的连线
			for i in range(len(save_intersection_xl) - 1):
				cv.line(img,
				        (int(save_intersection_xl[i]), int(save_intersection_yl[i])),
				        (int(save_intersection_xl[i + 1]), int(save_intersection_yl[i + 1])),
				        (0, 0, 255),
				        2)
			# 交点到最后一个端点
			cv.line(img,
			        (int(save_intersection_xl[-1]), int(save_intersection_yl[-1])),
			        (int(save_line_point_ex_l_list[-1]), int(save_line_point_ey_l_list[-1])),
			        (0, 0, 255),
			        2)

			# 上面偏移的线
			# 地点到第一个交点
			cv.line(img,
			        (int(save_line_point_sx_r_list[0]), int(save_line_point_sy_r_list[0])),
			        (int(save_intersection_xr[0]), int(save_intersection_yr[0])),
			        (0, 0, 255),
			        2)
			# 交点之间的连线
			for i in range(len(save_intersection_xr) - 1):
				cv.line(img,
				        (int(save_intersection_xr[i]), int(save_intersection_yr[i])),
				        (int(save_intersection_xr[i + 1]), int(save_intersection_yr[i + 1])),
				        (0, 0, 255),
				        2)
			# 交点到最后一个端点
			cv.line(img,
			        (int(save_intersection_xr[-1]), int(save_intersection_yr[-1])),
			        (int(save_line_point_ex_r_list[-1]), int(save_line_point_ey_r_list[-1])),
			        (0, 0, 255),
			        2)

		"""交点不存在"""
		if not save_intersection_xl:
			cv.line(img,
			        (int(save_line_point_sx_l_list[0]), int(save_line_point_sy_l_list[0])),
			        (int(save_line_point_ex_l_list[-1]), int(save_line_point_ey_l_list[-1])),
			        (0, 0, 255),
			        2)

			cv.line(img,
			        (int(save_line_point_sx_r_list[0]), int(save_line_point_sy_r_list[0])),
			        (int(save_line_point_ex_r_list[-1]), int(save_line_point_ey_r_list[-1])),
			        (0, 0, 255),
			        2)

		# 交点
		# for i in range(len(save_intersection_xl)):
		#     cv.circle(img, (int(save_intersection_xl[i]), int(save_intersection_yl[i])), 3, (255, 0, 0), -1)  # 上
		#     cv.circle(img, (int(save_intersection_xr[i]), int(save_intersection_yr[i])), 3, (255, 0, 0), -1)

		# 闭合首
		cv.line(img,
		        (int(save_line_point_sx_l_list[0]), int(save_line_point_sy_l_list[0])),
		        (int(save_line_point_sx_r_list[0]), int(save_line_point_sy_r_list[0])),
		        (0, 0, 255),
		        2)

		# 闭合尾
		cv.line(img,
		        (int(save_line_point_ex_l_list[-1]), int(save_line_point_ey_l_list[-1])),
		        (int(save_line_point_ex_r_list[-1]), int(save_line_point_ey_r_list[-1])),
		        (0, 0, 255),
		        2)

		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)  # 转为二值图
		_, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		# 画出轮廓线
		# draw_img = img.copy()
		# res = cv.drawContours(draw_img, contours, 1, (0, 255, 0), 2)
		# cv.imshow('res', res)

		# show_img("l", img)

		currentPoint_zoom = (
			int(currentPoint_zoom[0]),
			int(currentPoint_zoom[1])
		)

		cv.circle(img, currentPoint_zoom, 5, [0, 0, 255], -1)

		# It returns positive (inside), negative (outside), or zero (on an edge)
		dist = cv.pointPolygonTest(contours[1], currentPoint_zoom, False)
		# print("dist:", dist)

		BorderReminderLedXY = (w - 25, h - 18)  # 边界指示灯
		BorderReminderTextXY = (w - 320, h - 10)
		cv.circle(img, BorderReminderLedXY, 12, (0, 255, 0), -1)
		cv.putText(img, "BorderReminder", BorderReminderTextXY, cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
		self.BorderReminder.setText(" ")

		if dist == -1:
			cv.circle(img, BorderReminderLedXY, 12, (0, 0, 255), -1)  # 边界报警指示灯
			self.BorderReminder.setText("！！！ 超出边界 ！！！")

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
		self.nowXY.setText("(%.3f, %.3f)" % (nowX, nowY))

	def update(self):
		g_ui_threadLock.acquire()
		global x_min, y_min
		worked_flag = gl.get_value("worked_flag")
		if worked_flag:
			self.rightWindow(self.imgBar, self.__thread.get_msg_deep())
		g_start_x_list = gl.get_value('g_start_x_list')
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
		                int(current_y),
		                )

		# self.showNowXY((current_x - x_min) * 5, (current_y - y_min) * 5)
		self.showNowXY((current_x - x_min) * zoom_x + delta, (current_y - y_min) * zoom_x + delta)
		g_ui_threadLock.release()


if __name__ == "__main__":
	gl.gl_init()
	app = QApplication(sys.argv)

	gps_thread = threading.Thread(target=multi_thread.thread_gps_func, daemon=False)
	_4g_thread = threading.Thread(target=multi_thread.thread_4g_func, daemon=False)
	gyro_thread = threading.Thread(target=multi_thread.thread_gyro_func, daemon=True)
	g_laser1_thread = threading.Thread(target=multi_thread.thread_laser1_func, daemon=True)
	g_laser2_thread = threading.Thread(target=multi_thread.thread_laser2_func, daemon=True)
	g_laser3_thread = threading.Thread(target=multi_thread.thread_laser3_func, daemon=True)
	calculate_thread = threading.Thread(target=calculate.altitude_calculate_func, daemon=False)

	gps_thread.start()  # 启动线程
	_4g_thread.start()
	# gyro_thread.start()
	# g_laser1_thread.start()
	# # g_laser2_thread.start()
	# # g_laser3_thread.start()
	# sleep(1)
	# calculate_thread.start()

	mainWindow = MyWindows()

	while True:
		reced_flag = gl.get_value("reced_flag")
		if reced_flag:
			reced_flag = False
			gl.set_value("reced_flag", reced_flag)
			mainWindow = MyWindows()
			mainWindow.show()
			sys.exit(app.exec_())
