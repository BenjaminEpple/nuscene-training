import argparse
import subprocess
import sys

from nuscenes import NuScenes
from pynput import keyboard
from typing import List, Union

from viewer.worker import (
    SCENE_OPTION,
    ROOT_DIR,
    SENSOR_TYPE_CAMERA,
    TOP_RIGHT,
    TOP_LEFT,
    TOKEN_OPTION,
    WINDOW_POS_OPTION,
    SENSOR_TYPE_LIDAR_RADAR,
    SENSOR_TYPE_OPTION,
)


def get_next_sample_token(nusc, sample_token: str) -> Union[str, None]:
    sample = nusc.get("sample", sample_token)
    return sample.get("next", None)


def get_previous_sample_token(nusc, sample_token: str) -> Union[str, None]:
    sample = nusc.get("sample", sample_token)
    return sample.get("prev", None)


def parse_args():
    parser = argparse.ArgumentParser(description="NuScenes Viewer.")

    parser.add_argument(
        SCENE_OPTION,
        type=int,
        required=True,
        help="An integer option for demonstration purposes.",
    )
    args = parser.parse_args()
    # Validate the arguments
    if args.scene < 0 or args.scene > 10:
        print("Error: The scene number must be between 0 and 10.")
        parser.print_help()
        sys.exit(1)

    return args


class MasterViewer:
    def __init__(self, scene_number):
        self.nusc = NuScenes(
            version="v1.0-mini", dataroot=ROOT_DIR.as_posix(), verbose=True
        )
        self.scene_number = scene_number
        self.scene: dict = self.nusc.scene[scene_number]
        self.sample_token = self.scene["first_sample_token"]

        # child processes
        self.camera_window = None
        self.lidar_radar_window = None

        # show initial windows
        self.update_windows()

    def generate_worker_script_cmd(self, sensor_type: str) -> List[str]:
        window_pos = TOP_RIGHT
        if sensor_type == SENSOR_TYPE_CAMERA:
            window_pos = TOP_LEFT
        return [
            sys.executable,
            "-m",
            "viewer.worker",
            SCENE_OPTION,
            str(self.scene_number),
            SENSOR_TYPE_OPTION,
            sensor_type,
            TOKEN_OPTION,
            self.sample_token,
            WINDOW_POS_OPTION,
            window_pos,
        ]

    def on_press(self, key):
        try:
            if key == keyboard.Key.left:
                print("You pressed the left arrow key!")
                previous = get_previous_sample_token(self.nusc, self.sample_token)
                if previous is not None:
                    self.sample_token = previous
                    self.update_windows()
                else:
                    print("There is no previous sample token.")

            elif key == keyboard.Key.right:
                print("You pressed the right arrow key!")
                next = get_next_sample_token(self.nusc, self.sample_token)
                if next is not None:
                    self.sample_token = next
                    self.update_windows()
                else:
                    print("There is no next sample token.")

        except AttributeError:
            pass

    def update_windows(self):
        # kill processes if we are already running
        self.kill()

        # show new windows
        if self.camera_window is None and self.lidar_radar_window is None:
            self.camera_window = subprocess.Popen(
                self.camera_window_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.lidar_radar_window = subprocess.Popen(
                self.lidar_radar_window_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(
                f"Started processes:\n "
                f"Process 1: {' '.join(self.camera_window_cmd)}\n  "
                f"Process 2: {' '.join(self.lidar_radar_window_cmd)}"
            )

    @property
    def camera_window_cmd(self):
        return self.generate_worker_script_cmd(SENSOR_TYPE_CAMERA)

    @property
    def lidar_radar_window_cmd(self):
        return self.generate_worker_script_cmd(SENSOR_TYPE_LIDAR_RADAR)

    def kill(self):
        for proc in self.camera_window, self.lidar_radar_window:
            if proc is not None:
                proc.kill()
        self.camera_window = None
        self.lidar_radar_window = None


def main():
    args = parse_args()

    viewer = MasterViewer(args.scene)
    # Set up the listener
    listener = keyboard.Listener(on_press=viewer.on_press)
    listener.start()
    print("Press left or right arrow keys.")

    # Keep the script running to listen for key presses
    listener.join()


if __name__ == "__main__":
    main()
