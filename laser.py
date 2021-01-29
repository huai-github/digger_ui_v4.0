from bsp_serialport import SerialPortCommunication


class Laser(object):
	def __init__(self, com):
		self.distance = 0
		self.LASER_COM = com
		self.LASER_REC_BUF_LEN = 11
		self.com_laser = SerialPortCommunication(self.LASER_COM, 9600, 0.5)

	def get_distance(self):
		while True:
			laser_rec_buf = self.com_laser.read_size(self.LASER_REC_BUF_LEN)  # bytes
			# print(laser_rec_buf)
			# SUCC: 	b'\x80\x06\x83000.212\xa4'
			# ERROR: 	b'\x80\x06\x83ERR--15N'

			if laser_rec_buf != b'':
				# 切片有效数据
				# ADDR 06 83 3X 3X 3X 2E 3X 3X 3X CS
				distance = laser_rec_buf[3:10]
				# print(distance)	# b'000.103'
				if distance == b'ERR--15':
					print("--LASER READ ERROR--")
				else:
					self.distance = float(distance)
					# print("distance: ", self.distance)
				return self.distance

