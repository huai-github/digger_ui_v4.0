from math import sqrt
from time import sleep

from bsp_serialport import *
from bsp_4g import *
from bsp_gps import *
from datetime import datetime
from gyro import Gyro
import threading
import globalvar as gl
from laser import Laser

g_gps_threadLock = threading.Lock()
g_4g_threadLock = threading.Lock()
g_laser_threadLock = threading.Lock()
g_gyro_threadLock = threading.Lock()
g_laser1_threadLock = threading.Lock()
g_laser2_threadLock = threading.Lock()
g_laser3_threadLock = threading.Lock()

# 消息类型。1：心跳，2：上报
TYPE_HEART = 1
TYPE_SEND = 2
# 挖掘机ID
diggerId = 539081892037656576
# 高斯坐标，全局变量，double类型
g_x = 0
g_y = 0
g_h = 0

g_distance = 0
g_roll = 0
g_pitch = 0
g_yaw = 0

flag = False


class TimeInterval(object):
	def __init__(self, start_time, interval, callback_proc, args=None, kwargs=None):
		self.__timer = None
		self.__start_time = start_time
		self.__interval = interval
		self.__callback_pro = callback_proc
		self.__args = args if args is not None else []
		self.__kwargs = kwargs if kwargs is not None else {}

	def exec_callback(self, args=None, kwargs=None):
		self.__callback_pro(*self.__args, **self.__kwargs)
		self.__timer = threading.Timer(self.__interval, self.exec_callback)
		self.__timer.start()

	def start(self):
		interval = self.__interval - (datetime.now().timestamp() - self.__start_time.timestamp())
		# print( interval )
		self.__timer = threading.Timer(interval, self.exec_callback)
		self.__timer.start()

	def cancel(self):
		self.__timer.cancel()
		self.__timer = None


def thread_gps_func():
	GPS_COM = "com3"
	GPS_REC_BUF_LEN = 138
	values = []
	data_right_flag = None
	while True:
		gps_data = GPSINSData()
		gps_msg_switch = LatLonAlt()
		gps_rec_buffer = []
		gps_com = SerialPortCommunication(GPS_COM, 115200, 0.2)  # 5Hz
		gps_com.rec_data(gps_rec_buffer, GPS_REC_BUF_LEN)  # int
		gps_com.close_com()

		g_gps_threadLock.acquire()  # 加锁

		data_right_flag = gps_data.gps_msg_analysis(gps_rec_buffer)
		if data_right_flag:
			gps_msg_switch.latitude, gps_msg_switch.longitude, gps_msg_switch.altitude, \
				gps_msg_switch.yaw,  gps_msg_switch.yaw_state = gps_data.gps_typeswitch()
			print("纬度：%s\t经度：%s\t海拔：%s\t" % (gps_msg_switch.latitude, gps_msg_switch.longitude, gps_msg_switch.altitude))
			print("偏航角：", gps_msg_switch.yaw)

			"""经纬度转高斯坐标"""
			global g_x, g_y, g_h
			g_x, g_y = LatLon2XY(gps_msg_switch.latitude, gps_msg_switch.longitude)
			g_h = gps_msg_switch.altitude
			gl.set_value("gps_h", g_h)

			print("x:%s\ty:%s" % (g_x, g_y))
			# print("g_h:", g_h)

		else:
			print("数据错误\n")

		g_gps_threadLock.release()  # 解锁


def thread_4g_func():
	COM_ID_4G = "com5"
	rec = RecTasks()
	heart = Heart(TYPE_HEART, diggerId)
	com_4g = SerialPortCommunication(COM_ID_4G, 115200, 0.5)
	
	# 间隔一分钟发送一次心跳
	start = datetime.now().replace(minute=0, second=0, microsecond=0)
	minute = TimeInterval(start, 60, heart.send_heart_msg, [com_4g])
	minute.start()
	minute.cancel()

	while True:
		# 接收
		rec_buf = com_4g.read_line()  # byte -> bytes
		# print("rec_buf", rec_buf)
		g_4g_threadLock.acquire()  # 加锁
		worked_flag = gl.get_value("worked_flag")
		"""接收一次保存数据"""
		if rec_buf != b'':
			rec_buf_dict = task_switch_dict(rec_buf)
			rec.save_msg(rec_buf_dict)
			sx_list, sy_list, sh_list, sw_list, ex_list, ey_list, eh_list, ew_list = get_xyhw(rec_buf_dict)

			gl.set_value('g_start_x_list', sx_list)
			gl.set_value('g_start_y_list', sy_list)
			gl.set_value('g_start_h_list', sh_list)
			gl.set_value('g_start_w_list', sw_list)
			gl.set_value('g_end_x_list', ex_list)
			gl.set_value('g_end_y_list', ey_list)
			gl.set_value('g_end_h_list', eh_list)
			gl.set_value('g_end_w_list', ew_list)

			"""任务接收完成标志"""
			reced_flag = True
			gl.set_value("reced_flag", reced_flag)

		# 发送
		if worked_flag:
			h_o_min = gl.get_value("h_o_min")
			send = SendMessage(TYPE_HEART, diggerId, round(g_x, 3), round(g_y, 3), round(h_o_min, 3), 0)  # round(g_x, 3)保留3位小数
			send_msg_json = send.switch_to_json()
			com_4g.send_data(send_msg_json.encode('utf-8'))
			worked_flag = False
			gl.set_value("worked_flag", worked_flag)

		g_4g_threadLock.release()  # 解锁


def thread_gyro_func():
	GYRO_COM = "com32"
	gyro = Gyro()
	GYRO_REC_BUF_LEN = 33
	read_command = [0x50, 0x03, 0x00, 0x3d, 0x00, 0x03, 0x99, 0x86]
	com_gyro = SerialPortCommunication(GYRO_COM, 9600, 0.5)
	while True:
		com_gyro.send_data(read_command)
		gyro_rec_buf = com_gyro.read_size(GYRO_REC_BUF_LEN)

		RollH = gyro_rec_buf[3]
		RollL = gyro_rec_buf[4]
		PitchH = gyro_rec_buf[5]
		PitchL = gyro_rec_buf[6]

		if gyro_rec_buf[0] == 0x50 and gyro_rec_buf[1] == 0x03:
			g_gyro_threadLock.acquire()
			gyro.roll = int(((RollH << 8) | RollL)) / 32768 * 180
			gyro.pitch = int(((PitchH << 8) | PitchL)) / 32768 * 180

			gyro.roll = round(gyro.roll, 2) # 保存2位小数
			gyro.pitch = round(gyro.pitch, 2)

			if gyro.roll is not None and gyro.pitch is not None:
				gl.set_value("roll", gyro.roll)
				gl.set_value("pitch", gyro.pitch)
				# print("roll:", gyro.roll)
				# print("pitch:", gyro.pitch)

			g_gyro_threadLock.release()


def thread_gyro_3_func():
	GYRO_COM = "com4"
	gyro = Gyro()
	GYRO_REC_BUF_LEN = (11 * 4)
	com_gyro = SerialPortCommunication(GYRO_COM, 115200, 0.5)
	while True:
		g_gyro_threadLock.acquire()
		gyro_rec_buf = com_gyro.read_size(GYRO_REC_BUF_LEN)
		# print(gyro_rec_buf)
		target_index = gyro_rec_buf.find(0x53)  # 角度输出
		if target_index != (-1):
			if gyro_rec_buf[target_index - 1] == 0x55:  # 数据头
				data = gyro_rec_buf[(target_index - 1):(target_index + 10)]
				gyro.roll = int(((data[3] << 8) | data[2])) / 32768 * 180
				gyro.pitch = int(((data[5] << 8) | data[4])) / 32768 * 180
				gyro.yaw = int(((data[7] << 8) | data[6])) / 32768 * 180
				print("roll:", gyro.roll)
				print("pitch:", gyro.pitch)
				print("yaw:", gyro.yaw)
				print("------------------------------------------")
			else:
				print("header 1 error")
				return -1
		else:
			print("header 2 error")
			return -1
		g_gyro_threadLock.release()


def thread_laser1_func():
	LASER1_COM = "com37"
	laser1 = Laser(LASER1_COM)
	while True:
		g_laser1_threadLock.acquire()
		laser1_dist = laser1.get_distance()
		if laser1_dist is not None:
			gl.set_value("laser1_dist", laser1_dist)
			print("laser1_dist", laser1_dist)
		g_laser1_threadLock.release()


def thread_laser2_func():
	LASER2_COM = "com38"
	laser2 = Laser(LASER2_COM)
	while True:
		g_laser2_threadLock.acquire()
		laser2_dist = laser2.get_distance()
		if laser2_dist is not None:
			gl.set_value("laser2_dist", laser2_dist)
			print("laser2_dist", laser2_dist)

		g_laser2_threadLock.release()


def thread_laser3_func():
	LASER3_COM = "com39"
	laser3 = Laser(LASER3_COM)
	while True:
		g_laser3_threadLock.acquire()
		laser3_dist = laser3.get_distance()
		if laser3_dist is not None:
			gl.set_value("laser3_dist", laser3_dist)
			print("laser3_dist", laser3_dist)

		g_laser3_threadLock.release()
