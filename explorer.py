from typing import List

import numpy as np
from matplotlib import pyplot as plt
from nuscenes import NuScenesExplorer
from nuscenes.utils.geometry_utils import BoxVisibility


class CustomExplorer(NuScenesExplorer):
    def render_sample(
        self,
        token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        nsweeps: int = 1,
        out_path: str = None,
        show_lidarseg: bool = False,
        filter_lidarseg_labels: List = None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
    ) -> None:
        """
        Render all LIDAR and camera sample_data in sample along with annotations.
        :param token: Sample token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param show_lidarseg: Whether to show lidar segmentations labels or not.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to show the rendered sample in a window or not.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        record = self.nusc.get("sample", token)
        camera_data, lidar_data, radar_data = self.split_radar_lidar_vision(record)

        # Create plots.
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + len(camera_data) + num_lidar_plots
        cols = 2
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))

        # Plot radars into a single subplot.
        if len(radar_data) > 0:
            ax = axes[0, 0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.render_sample_data(
                    sd_token,
                    with_anns=i == 0,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    nsweeps=nsweeps,
                    verbose=False,
                )
            ax.set_title("Fused RADARs")

        # Plot lidar into a single subplot.
        if len(lidar_data) > 0:
            for (_, sd_token), ax in zip(
                lidar_data.items(), axes.flatten()[num_radar_plots:]
            ):
                self.render_sample_data(
                    sd_token,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    nsweeps=nsweeps,
                    show_lidarseg=show_lidarseg,
                    filter_lidarseg_labels=filter_lidarseg_labels,
                    lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                    verbose=False,
                    show_panoptic=show_panoptic,
                )

        # Plot cameras in separate subplots.
        for (_, sd_token), ax in zip(
            camera_data.items(), axes.flatten()[num_radar_plots + num_lidar_plots :]
        ):
            if show_lidarseg or show_panoptic:
                sd_record = self.nusc.get("sample_data", sd_token)
                sensor_channel = sd_record["channel"]
                valid_channels = [
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_LEFT",
                    "CAM_BACK",
                    "CAM_BACK_RIGHT",
                ]
                assert (
                    sensor_channel in valid_channels
                ), "Input camera channel {} not valid.".format(sensor_channel)

                self.render_pointcloud_in_image(
                    record["token"],
                    pointsensor_channel="LIDAR_TOP",
                    camera_channel=sensor_channel,
                    show_lidarseg=show_lidarseg,
                    filter_lidarseg_labels=filter_lidarseg_labels,
                    ax=ax,
                    verbose=False,
                    lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                    show_panoptic=show_panoptic,
                )
            else:
                self.render_sample_data(
                    sd_token,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    nsweeps=nsweeps,
                    show_lidarseg=False,
                    verbose=False,
                )

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)

        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()

    def split_radar_lidar_vision(self, record):
        # Separate RADAR from LIDAR and vision.
        radar_data = {}
        camera_data = {}
        lidar_data = {}
        for channel, token in record["data"].items():
            sd_record = self.nusc.get("sample_data", token)
            sensor_modality = sd_record["sensor_modality"]

            if sensor_modality == "camera":
                camera_data[channel] = token
            elif sensor_modality == "lidar":
                lidar_data[channel] = token
            else:
                radar_data[channel] = token
        return camera_data, lidar_data, radar_data

    def render_sample_lidar_radar(
        self,
        token: str,
        window_position: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        nsweeps: int = 1,
        out_path: str = None,
        show_lidarseg: bool = False,
        filter_lidarseg_labels: List = None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
    ) -> None:

        record = self.nusc.get("sample", token)
        camera_data, lidar_data, radar_data = self.split_radar_lidar_vision(record)

        # Create plots.
        print(f"Found {len(radar_data)} radar plots")
        print(f"Found {len(lidar_data)} lidar plots")
        num_radar_plots = 1 if len(radar_data) > 0 else 0
        num_lidar_plots = 1 if len(lidar_data) > 0 else 0
        n = num_radar_plots + num_lidar_plots
        cols = 2
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))

        # Plot radars into a single subplot.
        if len(radar_data) > 0:
            ax = axes[0]
            for i, (_, sd_token) in enumerate(radar_data.items()):
                self.render_sample_data(
                    sd_token,
                    with_anns=i == 0,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    nsweeps=nsweeps,
                    verbose=False,
                )
            ax.set_title("Fused RADARs")

        # Plot lidar into a single subplot.
        if len(lidar_data) > 0:
            for (_, sd_token), ax in zip(
                lidar_data.items(), axes.flatten()[num_radar_plots:]
            ):
                self.render_sample_data(
                    sd_token,
                    box_vis_level=box_vis_level,
                    ax=ax,
                    nsweeps=nsweeps,
                    show_lidarseg=show_lidarseg,
                    filter_lidarseg_labels=filter_lidarseg_labels,
                    lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                    verbose=False,
                    show_panoptic=show_panoptic,
                )

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.canvas.set_window_title('Radar and Lidar')

        # Position the figure window at the specified position
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry(
            window_position
        )


        if out_path is not None:
            plt.savefig(out_path)

        if verbose:
            plt.show()

    def render_cameras(
        self,
        token: str,
        window_position: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        nsweeps: int = 1,
        out_path: str = None,
        verbose: bool = True,
    ) -> None:
        record = self.nusc.get("sample", token)
        camera_data, _, _ = self.split_radar_lidar_vision(record)
        # Create plots.
        n = len(camera_data)
        cols = 3  # this was 2!
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 24))

        # Plot cameras in separate subplots.

        ordered_sd_tokens = [
            camera_data.get("CAM_FRONT_LEFT"),
            camera_data.get("CAM_FRONT"),
            camera_data.get("CAM_FRONT_RIGHT"),
            camera_data.get("CAM_BACK_LEFT"),
            camera_data.get("CAM_BACK"),
            camera_data.get("CAM_BACK_RIGHT"),
        ]

        for sd_token, ax in zip(ordered_sd_tokens, axes.flatten()):
            self.render_sample_data(
                sd_token,
                box_vis_level=box_vis_level,
                ax=ax,
                nsweeps=nsweeps,
                show_lidarseg=False,
                verbose=False,
            )

        # Change plot settings and write to disk.
        axes.flatten()[-1].axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.canvas.set_window_title('Cameras')

        if out_path is not None:
            plt.savefig(out_path)

        # Position the figure window at the specified position
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry(
            window_position
        )

        if verbose:
            plt.show()
