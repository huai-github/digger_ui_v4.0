import threading
from time import ctime, sleep


class MyThread(threading.Thread):
	def __init__(self, func, args, name='', daemon=False):
		""" 初始化线程功能类 """
		threading.Thread.__init__(self)  # 初始化父类
		self.func = func
		self.args = args
		self.name = name
		self.daemon = daemon
		self.res = 0
		self.result = False
		self.paused = True  # Start out paused.
		self.state = threading.Condition()

	def run(self):  # run为threading.Thread类中的线程功能方法, 继承时需要重写
		""" 本为构造类, 但在__init__中已经有了参数赋值
		这里是为功能函数赋值 """

		self.resume()
		while True:
			print("{} running ...\r".format(self.name), end="")
			self.result = False
			with self.state:
				if self.paused:
					print()
					print("{} paused ...".format(self.name))
					self.state.wait()  # Block execution until notified.

			# Do stuff.
			self.res = self.func(*self.args)  # *任意参数
			self.result = True

	def getResult(self):
		""" 获得运算结果 """
		return self.res

	def resume(self):
		with self.state:
			self.paused = False
			self.state.notify()  # Unblock self if waiting.

	def pause(self):
		with self.state:
			self.paused = True  # Block self.


if __name__ == "__main__":
	from time import sleep


	def func1():
		while True:
			print("This is ", func1.__name__)
			sleep(3)


	def func2():
		while True:
			print("This is ", func2.__name__)
			sleep(1)


	MyThread(func1, (), name='func1', daemon=True).start()
	MyThread(func2, (), name='func2', daemon=True).start()

	while True:
		pass
