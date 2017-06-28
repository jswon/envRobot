import sys
import pyueye
import urx
import pyGrip
import serial
import time
import numpy as np
import ur_safety
import pyueye
import Kinect_Snap
from util import *

# ----------------- Define -------------------------
PI = np.pi
HOME = (0, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
shf_way_pt = np.array([[-0.82222461061452856, -1.5587535549561358, -2.0142844897266556, -1.0569713950077662, 1.5327481491014201, 0.23544491609403506],
                       [-1.5591026208065346, -0.87423542232395968, 0.88383473320992845, -1.5660839378145119, 4.6582837735728653, 1.5947073375472192],
                       [-1.524196035766648, -0.79656827061021196, 1.0779153460316979, -1.8202038769048863, 4.6553167138444751, 1.5924384095196262],
                       [-1.535017077129013, -1.1815879036001613, 1.0709340290237206, -1.557706357404939, 4.6504297919388904, 1.5929620082952245],
                       [-1.4950490372583425, -1.5502014416213632, 1.0733774899765127, -1.3962634015954636, 4.6293113079897603, 1.6215854080279315],
                       [-1.5042992822939125, -1.2472122834751478, 1.073901088752111, -1.5341444125030159, 4.6266933141117681, 1.6214108751027323],
                       [-1.5100588688254937, -0.79656827061021196, 1.0751228192285072, -1.8051940453377351, 4.6223299909817817, 1.6214108751027323],
                       [-1.5334462808022178, -1.3646729421343662, 1.011069235680315, -1.7208946424664087, 4.6116834825446169, 1.5678292670665062]])
# --------------------------------------------------

class envRobot :
    def __init__(self, SOCKET_IP):
        # Connect
        self.gripper = pyGrip.gripper(host=SOCKET_IP)               # Gripper
        self.rob = urx.Robot(SOCKET_IP)                             # Robot
        self.safety = ur_safety.safety_chk(host= SOCKET_IP)   # Robot - Dashboard ( for collision check)
        self.bluetooth = serial.Serial("COM9", 9600, timeout=1)     # Tray
        self.local_cam = pyueye.uEyeCAM()
        self.global_cam = Kinect_Snap.take_snap()

        # Robot
        self.acc = 1.5; self.vel = 1.5
        self.isCollision = False

        print("Robot Environment Ready.", file=sys.stderr)

    def step(self, action):
        self.rob.movej(action, acc = self.acc, vel = self.vel)

        if self.collision_chk() == True :
            reward = 1234
            self.done = True
        # TODO : consider axis Z ????
        # elif self.maximum z threshold chk??????
        #    self.done = False
        #    reward = 1234

        else :
            reward = 1234
            self.done = True

        self.done = False
        self.state = self.get_state()

        """
        
        
        
        # TODO
        
        




        
        """

        reward = 1  #test

        return self.get_state(), reward, self.done

    def reset(self):
        # Reset state
        self.rob.movej(HOME)
        self.gripper.open()
        self.done = False
        self.isCollision = False

        # TODO : Data?

    def collision_chk(self):
        status = self.safety.status_chk()
        if status == "COLLISION":
            return True
        elif status == "NORMAL":
            return False

    def set_gripper(self, speed, force):
        self.gripper.set_gripper(speed, force)

    def gripper_close(self):
        # TODO Maybe value < THRESHOLD -> Grasp
        self.gripper.close()
        if self.gripper.DETECT_OBJ:
            #TODO
            k = 'DETECT_OBJ'

    def gripper_open(self):
        self.gripper.open()

    def shuffle_obj(self):
        acc = 2.5; vel = 2.5
        self.rob.movej(shf_way_pt[0], acc=acc, vel=vel)  # random
        self.rob.movej(shf_way_pt[1], acc=acc, vel=vel)
        self.gripper.move(104)                           # near handle open
        self.rob.movej(shf_way_pt[2], acc=acc, vel=vel)
        self.gripper.move(229)                           # handle grip
        self.rob.movej(shf_way_pt[3], acc=acc, vel=vel)
        self.rob.movej(shf_way_pt[4], acc=acc, vel=vel)
        time.sleep(2)                                    # stop delay 2~3 sec. # tray shuffle
        self.rob.movej(shf_way_pt[5], acc=acc, vel=vel)
        self.rob.movej(shf_way_pt[6], acc=acc, vel=vel)
        self.gripper.move(104)                           # near handle open
        self.rob.movej(shf_way_pt[7], acc=acc, vel=vel)

        # Vibrate tray, shuffle
        self.bluetooth.write("1".encode())
        time.sleep(3)  # waiting

    def get_state(self):
        global_img = self.global_cam.snap()
        local_img = self.local_cam.snap()

        # TODO 1 : Camera Image resizing

        # TODO 2 : Data
        rob_joint = self.rob.getj()
        tcp_pose = self.rob.getl()

        return [global_img, local_img, rob_joint, tcp_pose]

    def teaching_mode(self):
        # TODO : Reserved
        pass



