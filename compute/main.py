import json
import os
import sys
import cv2
import numpy as np
import cProfile
import re

from simple_term_menu import TerminalMenu

from calibrate_camera import calibrate_camera_new
from take_picture import take_picture
from detecte_green_square import get_center_green_cube
from compute_image_to_space import compute_image_to_space

def calibrate(log=False): # function processing the calibration of the two cameras
    """
    Calibrate the camera of the two raspberry pi
    :param log: If True, print the position matrix
    """

    # set orientation
    os.system("clear")
    options = ["0", "1", "2", "3"]
    terminal_menu = TerminalMenu(options, title="Mickey's orientation:")
    index = terminal_menu.show()
    mickey_index = int(index)

    os.system("clear")
    terminal_menu = TerminalMenu(options, title="minnie's orientation:")
    index = terminal_menu.show()
    minnie_index = int(index)

    os.system("clear")

    # each camera takes picture
    take_picture(raspberry_name="mickey", photo_name="photo_mickey.jpg")
    take_picture(raspberry_name="minnie", photo_name="photo_minnie.jpg")

    img_mickey = cv2.imread("/home/adrien/Bureau/Ned/compute/data/photos/photo_mickey.jpg", cv2.IMREAD_COLOR)
    img_minnie = cv2.imread("/home/adrien/Bureau/Ned/compute/data/photos/photo_minnie.jpg", cv2.IMREAD_COLOR)

    # calibrate each calibrate with the taken photos
    B_Mickey = calibrate_camera_new(img_mickey, index=mickey_index, log=log)
    B_minnie = calibrate_camera_new(img_minnie, index=minnie_index, log=log)

    return B_Mickey, B_minnie # return the found position and orientation of each camera

def get_point_position(B_Mickey, B_minnie, log=False):
    """
    Given the position of the two raspberry pi, compute the position of the point

    :param B_Mickey: The position matrix of the Mickey raspberry pi
    :param B_minnie: The position matrix of the minnie raspberry pi

    :return: the found position in a 4D vector
    """

    # take picture of the robot
    take_picture(raspberry_name="mickey", photo_name="photo_mickey.jpg")
    take_picture(raspberry_name="minnie", photo_name="photo_minnie.jpg")

    img_mickey = cv2.imread("/home/adrien/Bureau/Ned/compute/data/photos/photo_mickey.jpg", cv2.IMREAD_COLOR)
    img_minnie = cv2.imread("/home/adrien/Bureau/Ned/compute/data/photos/photo_minnie.jpg", cv2.IMREAD_COLOR)

    # find the position of the robot on each image
    center_minnie = get_center_green_cube(img_minnie, 1944, log)
    center_mickey = get_center_green_cube(img_mickey, 1944, log)

    # print(f"Center minnie: {center_minnie[0]-2592/2, 1944/2-center_minnie[1]}")
    # print(f"Center Mickey: {center_mickey[0]-2592/2, 1944/2-center_mickey[1]}")

    compute = True
    while compute:
        error = False
        try: 
            # compute the position of the arm given its position on each image
            position = compute_image_to_space(
                center_minnie[0]-2592/2,
                1944/2-center_minnie[1],
                1944/2,
                B_minnie,
                center_mickey[0]-2592/2,
                1944/2-center_mickey[1],
                1944/2,
                B_Mickey,
                log
            )
            compute = False
        except Exception as e: # can fail if not detecting the robot for example
            error = True
            options = ["Yes", "No"]
            terminal_menu = TerminalMenu(options, title="e\nRedo pictures ?")
            index = terminal_menu.show()
            if index == 1:
                return
        if log and not error: # visually verifying the picture and offer the possibility to redo it
            options = ["Yes", "No", "exit"]
            terminal_menu = TerminalMenu(options, title="Redo pictures ?")
            index = terminal_menu.show()
            if index == 2:
                return
            if not bool(index):
                os.system("clear")
                compute = True
            else:
                return position
    return position

def execute_program(file_path, B_Mickey, B_minnie, log=False): # open the program and automatically adds the calibration
    with open(file_path, 'r') as file:
        lines = file.readlines()
        code = 'import numpy as np\n'
        code += 'open("compute/logs/delta.log", "w").close()\n'
        code += 'open("compute/logs/movements.log", "w").close()\n'

        for line in lines:
            l = [w for w in line.split(' ') if w != '']
            if l != [] and l[0] == "#calibration": # 'calibration' means 'detection' in reality, the word is not well choosen
                var = l[1]
                if var[-1] in {'\n', '\r'}:
                    var = var[:-1]
                l = line.split(" ")
                n = [index for index in range(len(l)) if l[index] != ""][0]
                code += ' '*n + '__has_value = False\n'
                code += ' '*n + 'while not __has_value:\n'
                code += ' '*(n+3) + 'try:\n'
                code += ' '*(n+6) + '__delta_position_inserted = get_point_position(' # put the value of the position and orientation of each camera
                code += f"np.array([[{B_Mickey[0][0]}, {B_Mickey[0][1]}, {B_Mickey[0][2]}, {B_Mickey[0][3]}],"
                code += f"[{B_Mickey[1][0]}, {B_Mickey[1][1]}, {B_Mickey[1][2]}, {B_Mickey[1][3]}],"
                code += f"[{B_Mickey[2][0]}, {B_Mickey[2][1]}, {B_Mickey[2][2]}, {B_Mickey[2][3]}],"
                code += f"[{B_Mickey[3][0]}, {B_Mickey[3][1]}, {B_Mickey[3][2]}, {B_Mickey[3][3]}]]),"
                code += f"np.array([[{B_minnie[0][0]}, {B_minnie[0][1]}, {B_minnie[0][2]}, {B_minnie[0][3]}],"
                code += f"[{B_minnie[1][0]}, {B_minnie[1][1]}, {B_minnie[1][2]}, {B_minnie[1][3]}],"
                code += f"[{B_minnie[2][0]}, {B_minnie[2][1]}, {B_minnie[2][2]}, {B_minnie[2][3]}],"
                code += f"[{B_minnie[3][0]}, {B_minnie[3][1]}, {B_minnie[3][2]}, {B_minnie[3][3]}]]), log=" + str(log) + ")-"
                code += f"np.array({var}[:3]+[1])\n"
                code += ' '*(n+6) + 'print(f"Δx={__delta_position_inserted[0]*100:.2f}cm")\n' # prints the found difference
                code += ' '*(n+6) + 'print(f"Δy={__delta_position_inserted[1]*100:.2f}cm")\n'
                code += ' '*(n+6) + 'print(f"Δz={__delta_position_inserted[2]*100:.2f}cm")\n'
                code += ' '*(n+6) + 'print(f"||Δ||={np.linalg.norm(__delta_position_inserted)*100:.2f}cm")\n'
                code += ' '*(n+6) + 'with open ("compute/logs/delta.log", "a") as f:\n' # also adds it a log
                code += ' '*(n+9) + 'f.write(f"'
                code += '{__delta_position_inserted[0]} '
                code += '{__delta_position_inserted[1]} '
                code += '{__delta_position_inserted[2]}'
                code += '\\n")\n'
                code += ' '*(n+6) + '__has_value = True\n'
                code += ' '*(n+3) + 'except Exception as e:\n'
                code += ' '*(n+6) + 'print(e)\n'
            elif l != [] and ("ned.move_joints" in line or "ned.move_pose" in line) and l[0] != '#': # when moving the arm, write the ask position in a log
                l = line.split(" ")
                n = [index for index in range(len(l)) if l[index] != ""][0]
                code += line
                code += ' '*n + 'with open ("compute/logs/movements.log", "a") as f:\n'
                l = [s for s in re.split('[ ,]+', line) if s != '']
                l = l[0].split("([", maxsplit=1) + l[1:]
                index = l[-1].rfind("])")
                l = l[:-1] + [l[-1][:index]] + [l[-1][index+2:]]
                code += ' '*(n+6) + 'f.write(f"'
                code += str(l[0].split(".")[1])
                code += ' {' + str(l[1]) + '}'
                code += ' {' + str(l[2]) + '}'
                code += ' {' + str(l[3]) + '}'
                code += ' {' + str(l[4]) + '}'
                code += ' {' + str(l[5]) + '}'
                code += ' {' + str(l[6]) + '}'
                code += '\\n")\n'
            else:
                code += line
        with open('program_proof.py', 'w') as f: # saves the resulting program as a proof of what has been done
            f.write(code)
        exec(code, globals(), {})

def main_function(log=False):
    # proceed to a potential calibration before modifiying and executing the given program
    os.system("clear")
    options = ["No", "Yes", "exit"]
    terminal_menu = TerminalMenu(options, title="Proceed to a calibration before executing the program ?")
    index = terminal_menu.show()
    if index == 2:
        return
    calibration = bool(index)

    data = json.load(open("compute/position.json", "r"))
    B_Mickey = data["B1"]
    B_minnie = data["B2"]
    if calibration:
        B_Mickey, B_minnie = calibrate(log=log)
        for i in range(4):
            for j in range(4):
                data["B1"][i][j] = B_Mickey[i][j].astype(np.float64)
                data["B2"][i][j] = B_minnie[i][j].astype(np.float64)
        json.dump(data, open("compute/position.json", "w"), indent=4)
    B_Mickey = np.array(B_Mickey)
    B_minnie = np.array(B_minnie)
    
    if len(sys.argv) < 2:
        raise Exception("You must provide the path to the file to execute")
    path = sys.argv[1]

    execute_program(path, B_Mickey, B_minnie, log=log)

if __name__ == '__main__':
    main_function(log=False)
    # cProfile.run('main_function(log=True)',sort='tottime')