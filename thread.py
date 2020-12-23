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

# 消息类型。1：心跳，2：上报
TYPE_HEART = 1
TYPE_SEND = 2
# 挖掘机ID
diggerId = 531953905391632384
# 高斯坐标，全局变量，double类型
g_x = 0
g_y = 0
g_h = 0

g_distance = 0
g_roll = 0
g_pitch = 0
g_yaw = 0

g_worked_flag = False
g_reced_flag = False  # 任务接收完成标志
g_work_area = 0


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
	GPS_COM = "com5"
	GPS_REC_BUF_LEN = 138
	h_last = 0		# 上一次的海拔
	while True:
		gps_data = GPSINSData()
		gps_msg_switch = LatLonAlt()
		gps_rec_buffer = []
		gps_com = SerialPortCommunication(GPS_COM, 115200, 0.2)  # 5Hz
		gps_com.rec_data(gps_rec_buffer, GPS_REC_BUF_LEN)  # int
		gps_com.close_com()
		g_gps_threadLock.acquire()  # 加锁
		gps_data.gps_msg_analysis(gps_rec_buffer)
		gps_msg_switch.latitude, gps_msg_switch.longitude, gps_msg_switch.altitude = gps_data.gps_typeswitch()
		# print("纬度：%s\t经度：%s\t海拔：%s\t" % (gps_msg_switch.latitude, gps_msg_switch.longitude, gps_msg_switch.altitude))
		# 经纬度转高斯坐标
		global g_x, g_y, g_h
		g_x, g_y = LatLon2XY(gps_msg_switch.latitude, gps_msg_switch.longitude)
		g_h = gps_msg_switch.altitude
		g_gps_threadLock.release()  # 解锁
		"""判断挖完一次标志"""
		temp_h = g_h - h_last
		if temp_h:
			global g_worked_flag
			g_worked_flag = True
			h_last = g_h
		# print("x：%s\t y：%s\t deep：%s" % (g_x, g_y, g_h))  # 高斯坐标


def thread_4g_func():
	COM_ID_4G = "com21"
	rec = RecTasks()
	heart = Heart(TYPE_HEART, diggerId)
	com_4g = SerialPortCommunication(COM_ID_4G, 115200, 0.5)

	# 间隔一分钟发送一次心跳
	start = datetime.now().replace(minute=0, second=0, microsecond=0)
	# TODO:现在是6s，改成一分钟
	minute = TimeInterval(start, 6, heart.send_heart_msg, [com_4g])
	minute.start()
	minute.cancel()
	# sleep(3)
	while True:
		# 接收
		rec_buf = com_4g.read_line()  # byte -> bytes
		print("rec_buf", rec_buf)
		if rec_buf != b'':
			rec_buf_dict = task_switch_dict(rec_buf)
			rec.save_msg(rec_buf_dict)
			# print(rec.rec_task_dict["diggerId"])
			g_4g_threadLock.acquire()  # 加锁

			global g_work_area
			# gl.set_value('g_startX', rec.rec_task_dict["list"][g_work_area]["startX"])
			# gl.set_value('g_startY', rec.rec_task_dict["list"][g_work_area]["startY"])
			# gl.set_value('g_startH', rec.rec_task_dict["list"][g_work_area]["startH"])
			# gl.set_value('g_startW', rec.rec_task_dict["list"][g_work_area]["startW"])
			# gl.set_value('g_endX', rec.rec_task_dict["list"][g_work_area]["endX"])
			# gl.set_value('g_endY', rec.rec_task_dict["list"][g_work_area]["endY"])
			# gl.set_value('g_endH', rec.rec_task_dict["list"][g_work_area]["endH"])
			# gl.set_value('g_endW', rec.rec_task_dict["list"][g_work_area]["endW"])
			gl.set_value('g_startX', rec.rec_task_dict["list"][0]["startX"])
			gl.set_value('g_startY', rec.rec_task_dict["list"][0]["startY"])
			gl.set_value('g_startH', rec.rec_task_dict["list"][0]["startH"])
			gl.set_value('g_startW', rec.rec_task_dict["list"][0]["startW"])
			gl.set_value('g_endX', rec.rec_task_dict["list"][0]["endX"])
			gl.set_value('g_endY', rec.rec_task_dict["list"][0]["endY"])
			gl.set_value('g_endH', rec.rec_task_dict["list"][0]["endH"])
			gl.set_value('g_endW', rec.rec_task_dict["list"][0]["endW"])
			g_4g_threadLock.release()  # 解锁
			"""任务接收完成标志"""
			global g_reced_flag
			g_reced_flag = True

		# 发送
		send = SendMessage(TYPE_HEART, diggerId, round(g_x, 2), round(g_y, 2), round(g_h, 2), 0)
		send_msg_json = send.switch_to_json()
		com_4g.send_data(send_msg_json.encode('utf-8'))


def gyro_thread_func():
	GYRO_COM = "com9"
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
		g_gyro_threadLock.release()
