from math import acos, sqrt, cos

# 已知：laser_cd, laser_de, laser_fi,
# 通过激光传感器测得的距离计算角度
from bsp_gps import GPSINSData
from gyro import Gyro
from laser import Laser


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

# cos(angle_cab) = (((l_ac) ** 2) + ((l_ab) ** 2) - ((l_bc) ** 2)) / (2 * l_ac * l_ab)
# cos(angle_dge) = (((l_dg) ** 2) + ((l_eg) ** 2) - ((l_de) ** 2)) / (2 * l_dg * l_eg)
# cos(angle_fhi) = (((l_fh) ** 2) + ((l_hi) ** 2) - ((l_fi) ** 2)) / (2 * l_fh * l_hi)


def altitude_calculate_func():
	while True:
		laser1 = Laser("com11")
		l_bc = laser1.distance()
		laser2 = Laser("com11")
		l_de = laser2.distance()
		laser3 = Laser("com11")
		l_fi = laser3.distance()

		"""
		大臂：
		"""
		l_ab, l_ac, cag = 0
		gyro = Gyro()
		ab = gyro.pitch
		cab = laser_dist_to_angle(l_ab, l_ac, l_bc)
		ag = cab + ab - cag
		if ag > 0:
			print("上仰")
			pass
		else:
			print("下倾")
			pass

		"""
		小臂
		"""
		l_dg, l_ge, jgh, egh, dga = 0
		ga = -(90 - ag)
		agp = abs(ga)

		dge = laser_dist_to_angle(l_dg, l_ge, l_de)
		gj = 360 - jgh - egh - dga - agp - dge
		alpha = gj

		"""
		挖斗
		"""
		l_hj, l_hi, l_jk, l_ik, l_hf, fhi, ghf, ghj, gjf, kjo, gjh = 0
		jhi = 360 - ghj - gjf - fhi
		l_ij = sqrt((l_hi ** 2) + (l_hj ** 2) - (2 * l_hi * l_hj * cos(jhi)))

		hji = laser_dist_to_angle(l_hj, l_ij, l_hi)
		ijk = laser_dist_to_angle(l_jk, l_ij, l_ik)
		hjk = hji + ijk

		fhi = laser_dist_to_angle(l_hi, l_hf, l_fi)
		jhi = 360 - ghj - ghf - fhi

		jo = 360 + 180 + gj - (gjh + hjk + kjo)
		beta = jo
		"""
		高程计算
		已知：l_gj, l_jo
		"""
		l_gj, l_jo = 0
		gps = GPSINSData()
		hg = gps.gps_typeswitch()[-1]
		h0 = hg - l_gj * cos(alpha) - l_jo * cos(beta)

		return h0

