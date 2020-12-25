from ctypes import *
import cv2 as cv
import numpy as np
from math import sin, cos


def draw_all_line(img, sx_list, sy_list, ex_list, ey_list, l, circle_pt):
	for i in range(len(sx_list)):
		sx = sx_list[i]
		sy = sy_list[i]
		ex = ex_list[i]
		ey = ey_list[i]
		start_point = (sx, sy)
		end_point = (ex, ey)

		k = (ey - sy) / (ex - sx)
		theta = np.arctan(k)

		x_offset = l * sin(theta) * -1
		y_offset = l * cos(theta)
		M = np.float32([[1, 0, x_offset],
		                [0, 1, y_offset]])
		new_pt = cv.transform(np.float32([[start_point], [end_point]]), M).astype(np.int)

		x1_offset = l * sin(theta)
		y1_offset = l * cos(theta) * -1
		M1 = np.float32([[1, 0, x1_offset],
		                 [0, 1, y1_offset]])
		new_pt1 = cv.transform(np.float32([[start_point], [end_point]]), M1).astype(np.int)

		cv.line(img, start_point, end_point, (255, 0, 255), 2)

		box = np.array([tuple(new_pt1[0][0]), tuple(new_pt[0][0]), tuple(new_pt[1][0]), tuple(new_pt1[1][0])])
		cv.drawContours(img, [box[:, np.newaxis, :]], 0, (0, 0, 255), 2)

		cv.circle(img, circle_pt, 5, [0, 0, 255], -1)
		dist = cv.pointPolygonTest(box, circle_pt, False)

		if dist < 0:
			return dist


class TypeSwitchUnion(Union):
	_fields_ = [
		('double', c_double),
		('int', c_int),
		('float', c_float),
		('short', c_short),
		('char_2', c_char * 2),
		('char_8', c_char * 8)
	]
