import json


def task_switch_dict(rec_buf):
	rec_buf = rec_buf[0:-1]  # 去掉'\n'
	rec_to_str = str(rec_buf, encoding="utf-8")  # bytes -> str，不是dict！！！
	rec_buf_dict = eval(rec_to_str)  # str -> dict
	# print("rec_buf_dict:", rec_buf_dict)
	return rec_buf_dict


class Heart(object):
	def __init__(self, messageTypeId, diggerId):
		self.heart_dict = {
			"messageTypeId": messageTypeId,
			"diggerId": diggerId,  # 挖掘机ID
		}

	def send_heart_msg(self, com):
		send_buf_json = json.dumps(self.heart_dict)
		com.send_data(send_buf_json.encode('utf-8'))
		print("heart send succ\n")
	def rec_ack(self):
		pass


class SendMessage(object):
	def __init__(self, messageTypeId, diggerId, x, y, h, w):
		self.send_msg_dict = {
			"messageTypeId": messageTypeId,
			"diggerId": diggerId,
			"x": x,		# 单位为米，精确到厘米
			"y": y,
			"h": h,
			"w": w,
		}

	def switch_to_json(self):
		return json.dumps(self.send_msg_dict)


class RecTasks(object):
	def __init__(self):
		self.rec_task_dict = {
			"diggerId": 0,
			"list": [],
		}

	def save_msg(self, rec_buf_dict):
		self.rec_task_dict["diggerId"] = rec_buf_dict["diggerId"]
		self.rec_task_dict["list"] = rec_buf_dict["list"]


