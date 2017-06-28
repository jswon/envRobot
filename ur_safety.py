import sys
import socket
from threading import Thread, Condition, Lock
from time import sleep

# Robot safety checker Using Dashboard Server
class safety_chk(Thread):
    def __init__(self, host):
        Thread.__init__(self)
        self.BUFFSIZE = 1024
        self.Dashboard_socket = ''
        self.status = ''
        self._connect_server((host, 29999))  # (HOST : ROBOT IP, PORT : 29999 )  Dashboard server
        self.start()

    def _connect_server(self, ADDR):
        self.Dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.Dashboard_socket.settimeout(0.5)
        self.Dashboard_socket.connect(ADDR)
        chk = self._program_send("")
        print(chk, file=sys.stderr)
        self._program_send("close popup\n")

    def _program_send(self, cmd):
        self.Dashboard_socket.send(cmd.encode())
        return self.Dashboard_socket.recv(self.BUFFSIZE).decode("utf-8")  # received byte data to string

    def status_chk(self):
        # ~status chk reward, cur_angle, next_angle use ?

        robotmode = self._program_send("robotmode\n")[0:-1].split(' ')
        print(robotmode[1] is 'POWER_OFF')

        if robotmode[1] == 'POWER_OFF':
            self._program_send("power on\n")
            self._program_send("brake release\n")

        safetymode = self._program_send("safetymode\n")[0:-1].split(' ')

        print(safetymode)

        if safetymode[1] == "NORMAL":
            return "NORMAL"

        if safetymode[1] == "PROTECTIVE_STOP":
            self._program_send("unlock protective stop\n")
            return "COLLISION"  # collision
        if safetymode[1] == "SAFEGUARD_STOP":
            self._program_send("close safety popup")
            return "SAFEGUARD"

# cmd = "PolyscopeVersion" + "\n"
# cmd = cmd.encode()
# DashboardSocket.send(cmd)
# data = DashboardSocket.recv(BUFSIZE)
# print(data)


#cmd = "safetymode" + "\n"
#cmd = cmd.encode()

#DashboardSocket.send(cmd)
#data = DashboardSocket.recv(BUFSIZE)
#print(data)