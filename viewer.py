import argparse
import sys
import tkinter as tk
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes

from explorer import CustomExplorer

ROOT_DIR = Path(__file__).parent / "data" / "sets" / "nuscenes"


# Define constants for option keys
SCENE_OPTION = "--scene"
WINDOW_POS_OPTION = "--window-pos"
SENSOR_TYPE_OPTION = "--sensor-type"
TOKEN_OPTION = "--token"

# args
TOP_LEFT = "top-left"
TOP_RIGHT = "top-right"
SENSOR_TYPE_CAMERA = "camera"
SENSOR_TYPE_LIDAR_RADAR = "lidar-radar"


class Viewer:
    def __init__(
        self, nusc: NuScenes, initial_sample_token: str, explorer: CustomExplorer
    ):
        self.nusc = nusc
        self.sample_token = initial_sample_token
        self.explorer = explorer
        # Initialize counters
        self.left_count = 0
        self.right_count = 0

        # Tkinter window to capture key events
        self.root = tk.Tk()
        self.root.title("Key Press Captor")

        # Labels to show counters
        self.token_label = tk.Label(
            self.root,
            text=f"Showing token: {self.sample_token}",
            font=("Helvetica", 18),
        )
        self.token_label.pack(pady=10)

        # Bind key events to the Tkinter window
        self.root.bind("<Left>", self.left_key_press)
        self.root.bind("<Right>", self.right_key_press)

        # Create matplotlib windows for visual representation
        self.create_matplotlib_windows()

        # Start the Tkinter main loop
        self.root.mainloop()
        print("Exited main loop")

    def create_matplotlib_windows(self):
        record = self.nusc.get("sample", self.sample_token)
        camera_data, lidar_data, radar_data = self.explorer.split_radar_lidar_vision(
            record
        )

        # Left Counter Window
        self.fig_left, self.ax_left = plt.subplots()
        self.text_left = self.ax_left.text(
            0.5, 0.5, self._get_left_text(), ha="center", va="center", fontsize=20
        )
        self.ax_left.set_axis_off()
        self.fig_left.canvas.manager.set_window_title("Left Counter")

        # Right Counter Window
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + num_lidar_plots
        cols = 2
        self.fig_right, self.axes_right = plt.subplots(
            int(np.ceil(n / cols)), cols, figsize=(16, 24)
        )
        for ax in self.axes_right.flatten():
            ax.set_axis_off()
        self.fig_right.canvas.manager.set_window_title("Lidar and Radar")

        # Show the figures
        plt.show(block=False)

    def left_key_press(self, event):
        previous_sample_token = get_previous_sample_token(self.nusc, self.sample_token)
        if previous_sample_token is not None:
            self.sample_token = previous_sample_token
            self.increment_left()

    def right_key_press(self, event):
        next_sample_token = get_next_sample_token(self.nusc, self.sample_token)
        self.token_label.config(text=f"Showing token: {self.sample_token}")
        if next_sample_token is not None:
            self.sample_token = next_sample_token
            record = self.nusc.get("sample", self.sample_token)
            _, lidar_data, radar_data = self.explorer.split_radar_lidar_vision(record)

            self.explorer.render_sample_lidar_radar(
                self.sample_token,
                self.fig_right,
                self.axes_right,
                lidar_data,
                radar_data,
            )

    def _get_left_text(self):
        return f"Left: {self.left_count}"

    def _get_right_text(self):
        return f"Right: {self.sample_token}"

    def increment_left(self):
        self.left_count += 1
        self.token_label.config(text=f"Showing token: {self.sample_token}")
        self.text_left.set_text(self._get_left_text())
        self.fig_left.canvas.draw()


def get_next_sample_token(nusc, sample_token: str) -> Union[str, None]:
    sample = nusc.get("sample", sample_token)
    return sample.get("next", None)


def get_previous_sample_token(nusc, sample_token: str) -> Union[str, None]:
    sample = nusc.get("sample", sample_token)
    return sample.get("previous", None)


def get_screen_size() -> (int, int):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def get_right_hand_position(screen_width, screen_height):
    return f"{int(screen_width /2)}x{int(screen_height/2)}+{int(screen_width /2)}+0"


def get_left_hand_position(screen_width, screen_height):
    return f"{int(screen_width /2)}x{int(screen_height/2)}+0+0"


def parse_args():
    parser = argparse.ArgumentParser(description="NuScenes Viewer.")

    parser.add_argument(
        SCENE_OPTION,
        type=int,
        required=True,
        help="An integer option for demonstration purposes.",
    )

    parser.add_argument(
        WINDOW_POS_OPTION,
        type=str,
        choices=[TOP_LEFT, TOP_RIGHT],
        required=True,
        help=f"Specify the window position: '{TOP_LEFT}' or '{TOP_RIGHT}'.",
    )

    parser.add_argument(
        SENSOR_TYPE_OPTION,
        type=str,
        choices=[SENSOR_TYPE_CAMERA, SENSOR_TYPE_LIDAR_RADAR],
        required=True,
        help="Specify the sensor type: 'camera' or 'lidar-radar'.",
    )

    parser.add_argument(
        TOKEN_OPTION,
        type=str,
        required=False,
        help="Specify the token.",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Validate the arguments
    if args.scene < 0 or args.scene > 10:
        print("Error: The scene number must be between 0 and 10.")
        parser.print_help()
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args()

    width, height = get_screen_size()

    # get screen pos for figure
    if args.window_pos == TOP_LEFT:
        window_position = get_left_hand_position(width, height)
    elif args.window_pos == TOP_RIGHT:
        window_position = get_right_hand_position(width, height)
    else:
        print("Error: Not able to find screen position for figure.")
        sys.exit(1)

    nusc = NuScenes(version="v1.0-mini", dataroot=ROOT_DIR.as_posix(), verbose=True)

    my_scene: dict = nusc.scene[args.scene]
    explorer = CustomExplorer(nusc)

    # get screen sample token to render
    if not args.token:
        first_sample_token = my_scene["first_sample_token"]
    else:
        try:
            first_sample_token = my_scene[args.token]
        except AttributeError:
            print("Error: Did not find specified token in scene.")
            sys.exit(1)

    #  render data
    if args.sensor_type == SENSOR_TYPE_CAMERA:
        explorer.render_cameras(first_sample_token, window_position)
    elif args.sensor_type == SENSOR_TYPE_LIDAR_RADAR:
        explorer.render_sample_lidar_radar(first_sample_token, window_position)
