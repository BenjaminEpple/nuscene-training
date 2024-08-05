import sys
import time
from pathlib import Path

from nuscenes.nuscenes import NuScenes, NuScenesExplorer

NU_SCENES_DIR = Path(__file__).parent / "python-sdk" / "nuscenes"

ROOT_DIR = Path(__file__).parent / "data" / "sets" / "nuscenes"


class CustomExplorer(NuScenesExplorer):
    pass

if __name__ == "__main__":
    sys.path.append(NU_SCENES_DIR.as_posix())

    nusc = NuScenes(version="v1.0-mini", dataroot=ROOT_DIR.as_posix(), verbose=True)
    explorer = CustomExplorer(nusc)

    my_scene: dict = nusc.scene[0]
    number_of_samples: int = my_scene["nbr_samples"]

    first_sample_token = my_scene["first_sample_token"]
    sample = nusc.get("sample", first_sample_token)
    sample_counter = 1

    while sample_counter <= number_of_samples:
        next_sample_token = sample["next"]
        next_sample = nusc.get("sample", next_sample_token)
        explorer.render_sample(next_sample_token, show_panoptic=False)
        sample = next_sample
        time.sleep(0.5)
