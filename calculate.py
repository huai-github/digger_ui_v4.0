import threading
from math import acos, cos, pi, sin
import globalvar as gl

g_calculate_threadLock = threading.Lock()
h_o_min = 0


# def deep(dlt_bc, dlt_de, dlt_fi):
def deep(da_bi, xiao_bi, dou):
	# 各点的高程,h表示高程，g表示对应的点
	h_g = 0
	h_j = 0
	h_o = 0
	h_a = 119.5

	# pad笔记本下所测数据,l表示直线，_后面的两个字母表示对应的线段，a表示角度，dlt表示线段增量
	# l_bc = bc_tail + dlt_bc + bc_head
	# dlt_bc = 21.1 #l1
	# dlt_de = 81.4 #l2
	# dlt_fi = 52 #l3

	# 气缸总长度
	bc_all = da_bi      # 大臂
	de_all = xiao_bi    # 小臂
	fi_all = dou        # 斗

	dlt_bc = 0
	dlt_de = 0
	dlt_fi = 0

	l_ac = 146.4
	l_ab = 40.5
	l_ag = 300
	l_ad = 197.3
	l_bc = bc_all + dlt_bc
	l_cg = 177.8
	l_cd = 58.7
	l_dg = 172.6
	l_de = de_all + dlt_de
	l_eg = 49.1
	l_eh = 179.2
	l_fg = 37.4
	l_fh = 123
	l_fi = fi_all + dlt_fi
	l_gh = 135.6
	l_gj = 159
	l_hi = 33.6
	l_hj = 23.8
	l_ik = 32.5

	l_jk = 26.7
	l_jo = 94.1
	l_ko = 103.5

	# a表示角度，_后面的字母表示对应的角或射线
	# 大臂
	a_ab = -pi / 3 + 0.05
	# a_ab = -acos((99.3**2 + 41**2 - 93.5**2)/(2 * 99.3* 41))
	a_cab = acos((l_ac ** 2 + l_ab ** 2 - l_bc ** 2) / (2 * l_ac * l_ab))
	a_cag = acos((l_ac ** 2 + l_ag ** 2 - l_cg ** 2) / (2 * l_ac * l_ag))
	a_ag = a_cab + a_ab - a_cag
	print("a_ag", a_ag)

	# 小臂
	a_jgh = acos((l_gh ** 2 + l_gj ** 2 - l_hj ** 2) / (2 * l_gh * l_gj))
	a_egh = acos((l_eg ** 2 + l_gh ** 2 - l_eh ** 2) / (2 * l_eg * l_gh))
	a_dga = acos((l_dg ** 2 + l_ag ** 2 - l_ad ** 2) / (2 * l_dg * l_ag))
	a_dge = acos((l_dg ** 2 + l_eg ** 2 - l_de ** 2) / (2 * l_dg * l_eg))
	a_ga = - (pi / 2 - a_ag)
	a_agp = abs(a_ga)
	a_gj = 2 * pi - a_jgh - a_egh - a_dga - a_agp - a_dge
	print("a_gj:", a_gj)

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
	a_jo = a_jg - (a_gjh + a_hjk + a_kjo)
	print("a_jo", a_jo)
	print("(a_jo - 2 * pi) * 180 / pi :", (a_jo - 2 * pi) * 180 / pi)

	h_g = h_a + l_ag * sin(a_ag)
	print("h_g:", h_g)

	# 计算高程
	h_j = h_g - l_gj * cos(a_gj)
	h_o = h_j - l_jo * cos(a_jo)
	print("a_jo * 180 / pi :", a_jo * 180 / pi)
	print("h_j:", h_j)
	print("h_o:", h_o)

	return h_o


def altitude_calculate_func():
	global h_o_min
	while True:
		g_calculate_threadLock.acquire()

		h_g = gl.get_value("gps_h")  # gps高程
		# print("h_g", h_g)

		dlt_bc = gl.get_value("laser1_dist")
		dlt_de = gl.get_value("laser2_dist")
		dlt_fi = gl.get_value("laser3_dist")
		a_ab = gl.get_value("pitch")

		h_o = deep(144.5, 214.2, 152.7)

		"""判断挖完一次标志"""
		values.append(h_o)
		# values.append(g_h)
		# print("values", values)
		before_is_neg = False
		before_val = values[0]
		for i in range(1, len(values)):
			diff = values[i] - before_val
			if diff >= 0:
				if before_is_neg:
					worked_flag = True
					gl.set_value("worked_flag", worked_flag)  # 挖完一次
					h_o_min = values[i-1]
					# print("g_h", g_h)
					gl.set_value("h_o_min", h_o_min)  # 计算h0使用
					# print("***min***", values[i-1])

				before_is_neg = False
				before_val = values[i]
				values = []
			else:
				before_is_neg = True
				before_val = values[i]

		gl.set_value("h_o_min", h_o_min)
		print("h_o_min", h_o_min)

		g_calculate_threadLock.release()
