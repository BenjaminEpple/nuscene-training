import argparse
import sys
import tkinter as tk
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from explorer import CustomExplorer

ROOT_DIR = Path(__file__).parent.parent / "data" / "sets" / "nuscenes"


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


def get_screen_size() -> (int, int):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def get_right_hand_position(screen_width, screen_height):
    return f"{int(screen_width /2 - 100)}x{int(screen_height/2)}+{int(screen_width /2 + 100)}+0"


def get_left_hand_position(screen_width, screen_height):
    return f"{int(screen_width /2) - 100}x{int(screen_height/2)}+0+0"


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

    print(f"Using scene number {args.scene}.")
    my_scene: dict = nusc.scene[args.scene]

    explorer = CustomExplorer(nusc)

    # get screen sample token to render
    if not args.token:
        first_sample_token = my_scene["first_sample_token"]
    else:
        first_sample_token = args.token
    print(f"Using token: '{first_sample_token}'.")

    #  render data
    if args.sensor_type == SENSOR_TYPE_CAMERA:
        explorer.render_cameras(first_sample_token, window_position)
    elif args.sensor_type == SENSOR_TYPE_LIDAR_RADAR:
        explorer.render_sample_lidar_radar(first_sample_token, window_position)
