import os
import subprocess

def take_picture(
        raspberry_name: str="mickey",
        photo_name: str="photo_rasp.jpg"
    ) -> None:
    '''
    Take a picture with the raspberry pi camera and save it in the data/photos folder
    
    :param raspberry_name: The name of the raspberry pi (either 'mickey' or 'minnie')
    :param photo_name: The name of the photo to save
    '''

    if raspberry_name not in ["mickey", "minnie"]:
        raise ValueError("The raspberry name must be either 'mickey' or 'minnie'")
    
    try:
        subprocess.check_call(f"~/Bureau/photo.sh {raspberry_name}", shell=True)
        subprocess.check_call(f"mv ~/Bureau/photo.jpg ~/Bureau/Ned/compute/data/photos/{photo_name}", shell=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"{raspberry_name.capitalize()}'s camera is not connected")

if __name__ == '__main__':
    take_picture(raspberry_name="minnie")