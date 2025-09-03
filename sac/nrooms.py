# n_rooms_env.py
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

from nrooms_maps import (
    REGISTERED_LAYOUTS, WALL, FLOOR_CHARS,
    COLOR_WALL, COLOR_FLOOR, COLOR_AGENT, COLOR_GOAL,
    parse_map, list_floor_positions
)

def render_grid_rgb(grid: np.ndarray, pos: np.ndarray, goal: np.ndarray,
                    out_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    wall_mask = grid == WALL
    floor_mask = np.isin(grid, FLOOR_CHARS)
    img[wall_mask]  = COLOR_WALL
    img[floor_mask] = COLOR_FLOOR
    img[pos[0], pos[1]] = COLOR_AGENT
    img[goal[0], goal[1]] = COLOR_GOAL

    # Use OpenCV's resize with nearest-neighbor interpolation
    resized = cv2.resize(
        img,
        (out_size[1], out_size[0]),  # width, height
        interpolation=cv2.INTER_NEAREST
    )
    return resized

class NRoomsEnv(gym.Env):
    """
    Observation: Dict(image: HxWx3 uint8)
    Actions: Discrete(4): 0=up,1=down,2=left,3=right
    Ends on goal (terminated=True) or time limit (truncated=True).
    """
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        layout: str = "4_rooms",
        time_limit: int = 100,
        render_mode: Optional[str] = None,
        image_size: Tuple[int, int] = (64, 64),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if layout not in REGISTERED_LAYOUTS:
            raise ValueError(f"Unknown layout '{layout}' (choices: {list(REGISTERED_LAYOUTS.keys())})")

        self._layouts = [parse_map(m) for m in REGISTERED_LAYOUTS[layout]]
        self._time_limit = int(time_limit)
        self._image_size = image_size
        self.render_mode = render_mode
        self._np_random = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8),
        })

        self._t = 0
        self._grid: Optional[np.ndarray] = None
        self._pos:  Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._t = 0
        self._grid = self._np_random.choice(self._layouts)
        floor = list_floor_positions(self._grid)
        if floor.shape[0] < 2:
            raise RuntimeError("Layout needs ≥2 '.' cells for start/goal.")
        i0, i1 = self._np_random.choice(floor.shape[0], size=2, replace=False)
        self._pos  = floor[i0].copy()
        self._goal = floor[i1].copy()
        return self._get_obs(), {"position": self._pos.copy(), "goal": self._goal.copy()}

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._t += 1

        dr, dc = 0, 0
        if action == 0:   dr = -1   # up
        elif action == 1: dr = +1   # down
        elif action == 2: dc = -1   # left
        elif action == 3: dc = +1   # right

        nr, nc = int(self._pos[0] + dr), int(self._pos[1] + dc)
        h, w = self._grid.shape
        if 0 <= nr < h and 0 <= nc < w and self._grid[nr, nc] != WALL:
            self._pos[:] = (nr, nc)

        terminated = bool(np.all(self._pos == self._goal))
        truncated  = bool(self._t >= self._time_limit and not terminated)
        reward = -1.0
        return self._get_obs(), reward, terminated, truncated, {
            "position": self._pos.copy(), "goal": self._goal.copy()
        }

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"image": render_grid_rgb(self._grid, self._pos, self._goal, self._image_size)}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_obs()["image"]
        if self.render_mode == "ansi":
            out = self._grid.astype("U1").copy()
            rg, cg = int(self._goal[0]), int(self._goal[1])
            r,  c  = int(self._pos[0]),  int(self._pos[1])
            out[rg, cg] = "G"; out[r, c] = "A"
            return "\n".join("".join(row) for row in out)
        return None

def make_env(layout="4_rooms", time_limit=100, render_mode=None, image_size=(64, 64), seed=None) -> NRoomsEnv:
    return NRoomsEnv(layout=layout, time_limit=time_limit, render_mode=render_mode, image_size=image_size, seed=seed)

if __name__ == "__main__":
    env = make_env(layout="4_rooms", time_limit=50, render_mode="ansi")
    obs, info = env.reset()
    print(env.render())
    t = 0
    while True:
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        t += 1
        print("\nStep:", t, "Action:", a, "Reward:", r)
        print(env.render())
        if terminated or truncated:
            break
