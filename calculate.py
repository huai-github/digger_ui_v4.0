import threading
from math import acos, cos, pi
import globalvar as gl


g_calculate_threadLock = threading.Lock()

def laser_dist_to_angle(adjacent_1, adjacent_2, laser_dist):
	"""
	通过激光传感器测得的距离计算角度
	cos(angle) = （邻边平方和-对边平方）/（2*邻边积）
	:param adjacent_1: 邻边
	:param adjacent_2: 邻边
	:param laser_dist: 对边--激光传感器测得的距离
	:return: 角度angle
	"""
	return acos(((adjacent_1 ** 2) + (adjacent_2 ** 2) - (laser_dist ** 2)) / (2 * adjacent_1 * adjacent_2))


def altitude_calculate_func():
	while True:
		g_calculate_threadLock.acquire()
		# 各点的高程, h表示高程，g表示对应的点
		# h_g = 15
		h_g = gl.get_value("gps_h")
		h_j = 0
		h_o = 0

		# pad笔记本下所测数据, l表示直线，_后面的两个字母表示对应的线段, dlt表示线段增量
		dlt_bc = gl.get_value("laser1_dist")
		dlt_de = gl.get_value("laser2_dist")
		dlt_fi = gl.get_value("laser3_dist")
		a_ab = gl.get_value("pitch")

		# if h_g is not None and dlt_bc is not None and dlt_de is not None and dlt_fi is not None and a_ab is not None:
		if h_g is not None and dlt_bc is not None and a_ab is not None:
			print("h_g", h_g)
			# print("dlt_bc", dlt_bc)
			# print("a_ab", a_ab)
			l_ac = 7.5
			l_ab = 2.6
			l_ag = 15.2
			l_ad = 10.2
			l_bc = 6.2 + dlt_bc
			l_cg = 8.6
			l_cd = 2.7
			l_dg = 6
			# l_de = 7.5 + dlt_de
			l_de = 7.5000
			l_eg = 2.5
			l_eh = 7.1
			l_fg = 2.2
			l_fh = 5.7
			# l_fi = 5.6 + dlt_fi
			l_fi = 5.6000
			l_gh = 5
			l_gj = 6
			l_hi = 2.2
			l_hj = 1.0
			l_ik = 1.9
			l_ij = 2.4
			l_jk = 1.2
			l_jo = 4.0
			l_ko = 4.6

			# a表示角度，_后面的字母表示对应的角或射线
			# 大臂
			# a_ab = -0.09

			a_cab = acos((l_ac ** 2 + l_ab ** 2 - l_bc ** 2) / (2 * l_ac * l_ab))
			a_cag = acos((l_ac ** 2 + l_ag ** 2 - l_cg ** 2) / (2 * l_ac * l_ag))
			a_ag = a_cab + a_ab - a_cag

			# 小臂
			a_jgh = acos((l_gh ** 2 + l_gj ** 2 - l_hj ** 2) / (2 * l_gh * l_gj))
			a_egh = acos((l_eg ** 2 + l_gh ** 2 - l_eh ** 2) / (2 * l_eg * l_gh))
			a_dga = acos((l_dg ** 2 + l_ag ** 2 - l_ad ** 2) / (2 * l_dg * l_ag))
			a_dge = acos((l_dg ** 2 + l_eg ** 2 - l_de ** 2) / (2 * l_dg * l_eg))
			a_ga = -(pi / 2 - a_ag)
			a_agp = abs(a_ga)
			a_gj = 2 * pi - a_jgh - a_egh - a_dga - a_agp - a_dge

			# 挖斗
			# 计算角hjk
			a_ghj = acos((l_gh ** 2 + l_hj ** 2 - l_gj ** 2) / (2 * l_gh * l_hj))
			a_ghf = acos((l_gh ** 2 + l_fh ** 2 - l_fg ** 2) / (2 * l_gh * l_fh))
			a_fhi = acos((l_fh ** 2 + l_hi ** 2 - l_fi ** 2) / (2 * l_fh * l_hi))
			a_jhi = 2 * pi - a_ghj - a_ghf - a_fhi
			l_ij = (l_hi ** 2 + l_hj ** 2 - 2 * l_hi * l_hj * cos(a_jhi)) ** 0.5

			a_hji = acos((l_hj ** 2 + l_ij ** 2 - l_hi ** 2) / (2 * l_hj * l_ij))
			a_ijk = acos((l_ij ** 2 + l_jk ** 2 - l_ik ** 2) / (2 * l_ij * l_jk))
			a_hjk = a_hji + a_ijk

			# 计算jo
			a_jg = pi + a_gj
			a_gjh = acos((l_gj ** 2 + l_hj ** 2 - l_gh ** 2) / (2 * l_gj * l_hj))
			a_kjo = acos((l_jk ** 2 + l_jo ** 2 - l_ko ** 2) / (2 * l_jk * l_jo))
			a_jo = 2 * pi + a_jg - (a_gjh + a_hjk + a_kjo)
			# (a_jo - 2 * pi) * 180.0 / pi
			# print((a_jo - 2 * pi) * 180.0 / pi)

			# 计算高程
			h_j = h_g - l_gj * cos(a_gj)
			h_o = h_j - l_jo * cos(a_jo)
			gl.set_value("h_o", h_o)
			# print("h_o", h_o)
			g_calculate_threadLock.release()
