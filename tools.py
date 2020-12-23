from ctypes import *
from os.path import join
import cv2 as cv
import numpy as np
from math import sin, cos


def rad2angle(rad):
	return rad * 180 / np.pi


def get_line_angle(s, t):
	"""
	get_line_angle - 计算直线与 x 轴的夹角, 顺时针为正 0 - 360°
	@s:     起点
	@t:     终点
	@return:
		type - s 为原点时, t的相对象限
		angle - 夹角, 逆时针为正
	"""
	t = np.float32(t)
	s = np.float32(s)
	diff = t - s
	if diff.item(0) == 0:
		# 垂直
		diff.itemset(0, 1e-16 if t.item(0) > 0 else -1 * 1e-16)

	k = diff.item(1) / diff.item(0)
	angle = rad2angle(np.arctan(k))

	if angle > 0:  # 1, 3 象限 +90
		type = 1 if t.item(1) >= s.item(1) else 3

	elif angle == 0:  # x 负半轴给 3 象限, x 正半轴给 1 象限
		type = 3 if t.item(0) < s.item(0) else 1

	elif angle < 0:  # 2, 3 象限 象限区间左闭右开, 当 t 和 s 的 y 坐标相等时, 将其归为 3 象限
		type = 2 if t.item(1) > s.item(1) else 4

	# 4 象限: t 在 s 的右下方 [270, 360)
	# 3 象限: t 在 s 的左下方 [180, 270)
	# 2 象限: t 在 s 的左上方 [90, 180)
	# 1 象限: t 在 s 的右上方 [0, 90)
	if type == 2:
		angle += 180
	elif type == 3:
		angle += 180
	elif type == 4:
		angle += 360
	return type, angle


def work_area(img, a, b, l, point):
	k = (b[1] - a[1]) / (b[0] - a[0])
	theta = np.arctan(k)

	x_offset = l * sin(theta) * -1
	y_offset = l * cos(theta)
	M = np.float32([[1, 0, x_offset],
	                [0, 1, y_offset]])
	new_pt = cv.transform(np.float32([[a], [b]]), M).astype(np.int)

	x1_offset = l * sin(theta)
	y1_offset = l * cos(theta) * -1
	M1 = np.float32([[1, 0, x1_offset],
	                 [0, 1, y1_offset]])
	new_pt1 = cv.transform(np.float32([[a], [b]]), M1).astype(np.int)

	box = np.array([tuple(new_pt1[0][0]), tuple(new_pt[0][0]), tuple(new_pt[1][0]), tuple(new_pt1[1][0])])
	# cv.drawContours(img, [box[:, np.newaxis, :]], 0, (255, 0, 255), 2)

	circle_pt = np.array(point)
	# cv.circle(img, tuple(circle_pt.astype(np.int)), 2, (255, 0, 255), 2)

	# 旋转矩形框至垂直
	_, angle = get_line_angle(a, b)
	# print("before: ", angle)
	if angle <= 180:
		angle -= 90
	else:
		angle = 270 - angle
	# print("after: ", angle)
	M = cv.getRotationMatrix2D(tuple(np.mean(box, axis=0)), angle, 1)

	box = np.vstack([box, circle_pt])
	box = cv.transform(box[:, np.newaxis, :], M)
	# print(box)
	cv.drawContours(img, [box[:-1].astype(np.int)], -1, (0, 0, 255), 2)
	cv.circle(img, tuple(box[-1][0].astype(np.int)), 2, (0, 0, 255), 2)
	return box


class TypeSwitchUnion(Union):
	_fields_ = [
		('double', c_double),
		('int', c_int),
		('float', c_float),
		('short', c_short),
		('char_2', c_char * 2),
		('char_8', c_char * 8)
	]
