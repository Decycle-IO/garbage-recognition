import serial
import time

class CMDS:
    METAL= '1'
    PLASTIC= '2'
    OTHER= '3'
    OPEN_DOOR = '4'
    CLOSE_DOOR = '5'

class TrashCom:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate

    def send_cmd(self, cmd):
        with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
            ser.write(cmd.encode())


