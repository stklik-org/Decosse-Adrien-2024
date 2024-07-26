import numpy as np
import pyniryo
import os
import time

from simple_term_menu import TerminalMenu

loop = True

def test():
    global ned
    ned.move_joints([0, 0, 0, 0, 0, 0])
    ned.move_joints([0, 0, 0, 0, 0, -np.pi/2])
    ned.move_joints([0, 0, 0, 0, 0, 0])

def detect_green():
    theorical_position = [0, 0, 0]
    #calibration theorical_position

def loop_calibration():
    global ned
    os.system("clear")
    n = int(input("Number of loops: "))
    for _ in range(n):
        ned.calibrate_auto()

def go_to():
    global ned
    os.system("clear")
    j1 = float(input("J1: "))
    j2 = float(input("J2: "))
    j3 = float(input("J3: "))
    j4 = float(input("J4: "))
    j5 = float(input("J5: "))
    j6 = float(input("J6: "))
    ned.move_joints([j1, j2, j3, j4, j5, j6])
    theorical_position = [ned.get_pose().x, ned.get_pose().y, ned.get_pose().z]
    #calibration theorical_position

def make_loop():
    global ned
    epsilon = 0.2
    os.system("clear")
    n = int(input("Number of loops: "))
    alpha = 0
    # ned.move_joints([alpha, -np.pi/4, np.pi/4, 0, 0, 0])
    ned.move_joints([np.pi/4, 0, -np.pi/4, 0, np.pi/4, -np.pi/2])
    for _ in range(n):
        alpha = -np.pi/6
        ned.move_joints([np.pi/4, 0, -np.pi/4, 0, np.pi/4, -np.pi/2])
        while alpha < np.pi/2:
            alpha += epsilon
            ned.move_joints([np.pi/4, 0, -np.pi/4, 0, np.pi/4, -np.pi/2])
            ned.move_joints([np.pi/4, 0, -np.pi/4, 0, np.pi/4, np.pi/2])
            ned.move_joints([-np.pi/4, 0, -np.pi/4, 0, np.pi/4, np.pi/2])
            ned.move_joints([np.pi/4, 0, -np.pi/4, 0, np.pi/4, -np.pi/2])
        # ned.move_joints([0, -np.pi/4, np.pi/4, 0, 0, 0])
        # import time
        # time.sleep(0.5)
        # ned.move_joints([0, 0.6, -1.33, 0, 0, -0.01])
        # ned.move_joints([0, 0.6, 0, 0, 0, -0.01])
        theorical_position = [ned.get_pose().x, ned.get_pose().y, ned.get_pose().z]
        # calibration theorical_position

def test_camera():
    ned.move_joints([0, 0, 0, 0, 0, 0])
    n = int(input("Number of tests: "))
    for _ in range(n):
        theorical_position = [ned.get_pose().x, ned.get_pose().y, ned.get_pose().z]
        #calibration theorical_position

def exit_program():
    global loop
    loop = False

try:
    ned = pyniryo.NiryoRobot("169.254.200.200")
    ned.calibrate_auto()
except:
    ned = None
    print("Niryo One not connected")
    input("Press enter to continue")

while loop:
    os.system("clear")
    functions = [test, make_loop, loop_calibration, go_to, test_camera, ned.open_gripper, ned.close_gripper, ned.go_to_sleep, exit_program]
    # functions = [detect_green, exit_program]
    terminal_menu = TerminalMenu([f.__name__ for f in functions], title="Choose an action")
    index = terminal_menu.show()
    functions[index]()