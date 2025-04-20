import time
import serial
from pymodbus.client import ModbusTcpClient


class MyGripperSoftHand:

	TIME_ACTUATE	= time.time()
	TIME_PROTECTION	= 0.1

	def __init__(self) -> None:
		pass

	def my_init(self, port:str, baudrate:int, timeout=1, actuate_time_protection=0.5):

		# PAKETSEND [INFO-mini],[con-1],[con-2handsheck],[Act-1],[close-hand],[open-hand],[Deactive]
		self._PACKAGE_ = [0x3A,0x3A,0x01,0x04,0x0c,0x00,0x00,0x0c],[0x3A,0x3A,0x01,0x04,0x06,0x00,0x00,0x06],[0x3A,0x3A,0x01,0x02,0x81,0x81],[0x3A,0x3A,0x01,0x03,0x80,0x03,0x83],[0x3A,0x3A,0x01,0x04,0x82,0x4A,0x38,0xF0],[0x3A,0x3A,0x01,0x04,0x82,0x00,0x00,0x82],[0x3A,0x3A,0x01,0x03,0x80,0x00,0x80]

		self._SERIAL_	= serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
		for id in range(1, 100):
			packet = [0x3A,0x3A,id,0x02,0x00,0x00]
			self._SERIAL_.write(serial.to_bytes(packet))
			res	= self._SERIAL_.read_all()

			if res != '' :
			
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[1]))
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[2]))
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[3]))
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[2]))

				return True
			
			else:

				time.sleep(0.1)

		self.TIME_PROTECTION	= actuate_time_protection

		return False

	def my_get_info(self):

		packet = self._PACKAGE_[0]
		self._SERIAL_.write(serial.to_bytes(packet))
		res		= self._SERIAL_.read(1605)
		# res		= self._SERIAL_.read_all()
		text	= res.decode('ascii', 'ignore')
		
		print(text)

	def my_hand_open(self) -> bool:

		# limit fast actuation
		time_dv	= time.time() - self.TIME_ACTUATE
		print(time_dv)
		if time_dv <= self.TIME_PROTECTION:
			print(f"SoftHand open too fast ({time_dv} s.)")
			return True

		# Actuate
		for x in range(0, 3):

			try:
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[2]))
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[5]))
				self.TIME_ACTUATE	= time.time()
				return True

			except:
				print("Retry open SoftHand " + str(x) + " ...")
				time.sleep(0.2)

		return False

	def my_hand_close(self) -> bool:

		# limit fast actuation
		time_dv	= time.time() - self.TIME_ACTUATE
		if time_dv <= self.TIME_PROTECTION:
			print(f"SoftHand close too fast ({time_dv} s.)")
			return True

		# Actuate
		for x in range(0, 3):

			try:
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[2]))
				self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[4]))
				self.TIME_ACTUATE	= time.time()
				return True

			except:
				print("Retry close SoftHand " + str(x) + " ...")
				time.sleep(0.2)

		return False

	def my_get_position(self) -> str:

		packet = self._PACKAGE_[0]
		self._SERIAL_.write(serial.to_bytes(packet))
		res		= self._SERIAL_.read(1605)
		text	= res.decode('ascii', 'ignore')

		return	text

	def my_release(self):

		self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[6]))
		self._SERIAL_.write(serial.to_bytes(self._PACKAGE_[2]))
		self._SERIAL_.close()


class MyGripper3Finger:

	TIME_ACTUATE	= time.time()
	TIME_PROTECTION	= 0.5

	def __init__(self) -> None:
		pass

	def my_init(self, host:str, port:int, actuate_time_protection=0.5):

		self._MODBUS_	= ModbusTcpClient(host=host, port=port)
		
		if self._MODBUS_.connect() == False:
			return False
		
		self._MODBUS_.write_registers(0, [256, 0, 0])

		self.TIME_PROTECTION	= actuate_time_protection

		return True

	def my_hand_open(self) -> bool:

		# limit fast actuation
		time_dv	= time.time() - self.TIME_ACTUATE
		if time_dv <= self.TIME_PROTECTION:
			print(f"3-Finger open too fast ({time_dv} s.)")
			return True

		# Actuate
		for x in range(0, 3):

			try:
				self._MODBUS_.write_registers(0, [2816, 0, 51240])
				self.TIME_ACTUATE	= time.time()
				return True

			except:
				print("Retry open 3-Finger " + str(x) + " ...")
				time.sleep(0.2)

		return False

	def my_hand_close(self) -> bool:

		# limit fast actuation
		time_dv	= time.time() - self.TIME_ACTUATE
		if time_dv <= self.TIME_PROTECTION:
			print(f"3-Finger close too fast ({time_dv} s.)")
			return True

		# Actuate
		for x in range(0, 3):

			try:
				self._MODBUS_.write_registers(0, [2816, 100, 51240])
				self.TIME_ACTUATE	= time.time()
				return True

			except:
				print("Retry close 3-Finger " + str(x) + " ...")
				time.sleep(0.2)

		return False

	def my_get_position(self) -> str:

		# Get info
		# self._MODBUS_.write_registers(0, [2816, 100, 51240])

		return False

	def my_release(self):

		self._MODBUS_.close()
