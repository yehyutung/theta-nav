from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mujoco
import numpy as np


Obstacle = tuple[float, float, float, float]


@dataclass(frozen=True)
class WorldConfig:
    room_half_size: float = 5.0
    timestep: float = 0.01
    agent_radius: float = 0.15
    agent_height: float = 0.1
    wall_thickness: float = 0.1
    wall_height: float = 0.5
    max_linear_speed: float = 1.5
    max_turn_speed: float = 1.5
    linear_kv: float = 10.0
    turn_kv: float = 5.0
    include_agent_camera: bool = True
    include_top_camera: bool = True
    render_height: int = 96
    render_width: int = 96


def _obstacles_xml(obstacles: Iterable[Obstacle]) -> str:
    blocks: list[str] = []
    for i, (ox, oy, sx, sy) in enumerate(obstacles):
        blocks.append(
            f"""
    <geom name="obs_{i}" type="box"
          size="{sx} {sy} 0.5"
          pos="{ox} {oy} 0.5"
          rgba="0.6 0.2 0.2 1"/>"""
        )
    return "".join(blocks)


def build_world_xml(config: WorldConfig, obstacles: Optional[Iterable[Obstacle]] = None) -> str:
    obstacles = obstacles or []
    room = config.room_half_size
    wt = config.wall_thickness
    wh = config.wall_height

    cams = []
    if config.include_top_camera:
        cams.append('<camera name="top_down" pos="0 0 12" xyaxes="1 0 0 0 1 0" fovy="60"/>')

    if config.include_agent_camera:
        agent_camera = '<camera name="agent_cam" pos="0 0 0.2" xyaxes="1 0 0 0 0 1" fovy="90"/>'
    else:
        agent_camera = ""

    return f"""
<mujoco>
  <option timestep="{config.timestep}"/>
  <worldbody>
    <light diffuse=".6 .6 .6" pos="0 0 6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="{room} {room} 0.1" rgba="0.85 0.85 0.85 1"/>

    <geom name="wall_n" type="box" size="{room} {wt} {wh}" pos="0 {room} {wh}" rgba="0.25 0.25 0.25 1"/>
    <geom name="wall_s" type="box" size="{room} {wt} {wh}" pos="0 -{room} {wh}" rgba="0.25 0.25 0.25 1"/>
    <geom name="wall_e" type="box" size="{wt} {room} {wh}" pos="{room} 0 {wh}" rgba="0.25 0.25 0.25 1"/>
    <geom name="wall_w" type="box" size="{wt} {room} {wh}" pos="-{room} 0 {wh}" rgba="0.25 0.25 0.25 1"/>
    {_obstacles_xml(obstacles)}

    <body name="agent" pos="0 0 {config.agent_radius}">
      <joint name="agent_x" type="slide" axis="1 0 0"/>
      <joint name="agent_y" type="slide" axis="0 1 0"/>
      <joint name="agent_yaw" type="hinge" axis="0 0 1"/>
      <geom name="agent_geom" type="cylinder" size="{config.agent_radius} {config.agent_height}" rgba="0 0.5 1 1"/>
      {agent_camera}
    </body>
    {''.join(cams)}
  </worldbody>

  <actuator>
    <velocity name="vx" joint="agent_x" kv="{config.linear_kv}"/>
    <velocity name="vy" joint="agent_y" kv="{config.linear_kv}"/>
    <velocity name="yaw_rate" joint="agent_yaw" kv="{config.turn_kv}"/>
  </actuator>
</mujoco>
"""


class MujocoNavWorld:
    def __init__(self, config: Optional[WorldConfig] = None, obstacles: Optional[list[Obstacle]] = None):
        self.config = config or WorldConfig()
        self.obstacles = obstacles or []
        self.model = mujoco.MjModel.from_xml_string(build_world_xml(self.config, self.obstacles))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=self.config.render_width, height=self.config.render_height)

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> tuple[float, float, float]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = yaw
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.pose

    @property
    def pose(self) -> tuple[float, float, float]:
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])

    def step(self, forward_speed: float, turn_speed: float) -> tuple[float, float, float]:
        x, y, yaw = self.pose
        fwd = float(np.clip(forward_speed, -self.config.max_linear_speed, self.config.max_linear_speed))
        turn = float(np.clip(turn_speed, -self.config.max_turn_speed, self.config.max_turn_speed))

        self.data.ctrl[0] = fwd * np.cos(yaw)
        self.data.ctrl[1] = fwd * np.sin(yaw)
        self.data.ctrl[2] = turn
        mujoco.mj_step(self.model, self.data)
        return self.pose

    def render_agent_view(self) -> np.ndarray:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "agent_cam") < 0:
            raise ValueError("agent_cam was not included in this world config.")
        self.renderer.update_scene(self.data, camera="agent_cam")
        return self.renderer.render()

    def render_topdown_view(self) -> np.ndarray:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_down") < 0:
            raise ValueError("top_down camera was not included in this world config.")
        self.renderer.update_scene(self.data, camera="top_down")
        return self.renderer.render()

    def raycast_distance(
        self,
        x: float,
        y: float,
        yaw: float,
        angle_offset: float,
        max_range: float = 3.0,
    ) -> float:
        angle = yaw + angle_offset
        origin = np.array([x, y, 0.2], dtype=np.float64)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, origin, direction, None, 1, -1, geom_id)
        if distance <= 0.0:
            return max_range
        return float(min(distance, max_range))

    @property
    def num_contacts(self) -> int:
        return int(self.data.ncon)

    @property
    def is_in_collision(self) -> bool:
        return self.num_contacts > 0
