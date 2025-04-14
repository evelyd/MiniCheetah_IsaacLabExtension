import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.managers.command_manager import CommandManager, CommandTerm

from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import MiniCheetahModelEnv

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from .latent_model_env import MiniCheetahModelEnv

@configclass
class ReferenceVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

class ReferenceVelocityCommand(UniformVelocityCommand):
    """
    A CommandTerm that allows direct modification of the command.
    """

    def __init__(self, cfg: ReferenceVelocityCommandCfg, env: MiniCheetahModelEnv):
        super().__init__(cfg, env)
        self._mutable_command = torch.vstack([env.ref_trajectories[env.current_timesteps[i], i, 9:12].squeeze() for i in range(env.current_timesteps.shape[0])]) #initialize the command

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, command_dim)."""
        return self._mutable_command

    def compute(self, dt: float, env: MiniCheetahModelEnv):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command(env)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        pass

    def _update_command(self, env: MiniCheetahModelEnv):
        """Post-processes the velocity command.

        This function sets velocity command to that of the current timestep in the reference data.
        """
        # Take the velocity command directly from the reference data, using the current timesteps
        self._mutable_command = torch.vstack([
            env.ref_trajectories[env.current_timesteps[i], i, 9:12].squeeze()
            for i in range(env.current_timesteps.shape[0])
        ])
        print("mutable command:", self._mutable_command.shape)
        self.vel_command_b = self._mutable_command

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        print(f"command {self.command[:, :2]} and robot vel {self.robot.data.root_lin_vel_b[:, :2]}")
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

class MutableCommandManager(CommandManager):
    """Manager for allowing mutable commands.

    The command manager is used to generate commands for an agent to execute. It makes it convenient to switch
    between different command generation strategies within the same environment. For instance, in an environment
    consisting of a quadrupedal robot, the command to it could be a velocity command or position command.
    By keeping the command generation logic separate from the environment, it is easy to switch between different
    command generation strategies.

    The command terms are implemented as classes that inherit from the :class:`CommandTerm` class.
    Each command generator term should also have a corresponding configuration class that inherits from the
    :class:`CommandTermCfg` class.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        return super().__str__()

    """
    Operations.
    """

    def compute(self, dt: float, env: ManagerBasedRLEnv):
        """Updates the commands.

        This function calls each command term managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        # iterate over all the command terms
        for term in self._terms.values():
            # compute term's value
            term.compute(dt, env)