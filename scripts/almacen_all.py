import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class WarehouseEnv(gym.Env):
    def __init__(self, just_pick: bool = False, random_objects: bool = False):
        super(WarehouseEnv, self).__init__()

        # Define the size of the warehouse
        self.width = 10.0
        self.height = 10.0

        # Define the env mode
        self.just_pick = just_pick
        self.random_objects = random_objects

        # Define action and observation space
        if self.just_pick:
            # Up, Down, Left, Right, Pick
            self.action_space = spaces.Discrete(5)
        else:
            # Up, Down, Left, Right, Pick, Drop
            self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(11,),
            dtype=np.float32
        )

        # Define the shelves (x, y, width, height)
        self.shelves = [
            (1.9, 1.0, 0.2, 5.0),
            (4.9, 1.0, 0.2, 5.0),
            (7.9, 1.0, 0.2, 5.0)
        ]

        # Define the delivery area
        self.delivery_area = (2.5, 9, 5.0, 2.0)  # x, y, width, height

        # Define agent properties
        self.agent_radius = 0.2
        self.agent_velocity = 0.25
        self.pickup_distance = 0.6

        # Add a variable to store the figure and axis for rendering
        self.fig = None
        self.ax = None

        # Initialize the state
        self.reset()

    def reset(self):
        # Reset agent position
        self.agent_pos = self._get_random_empty_position()

        # Reset object positions
        self.object_positions = []
        randomize_shelves = self.random_objects
        if randomize_shelves:
            for shelf in self.shelves:
                pos = self._get_random_position_on_shelf(shelf)
                self.object_positions.append(pos)
        else:
            self.object_positions = [(2, 3.0), (8, 4.0), (5, 2.0)]

        # Reset agent's inventory
        self.agent_has_object = False

        # Reset delivery flag
        self.delivery = False

        # Reset collision flag
        self.collision = False

        return self._get_obs()

    def step(self, action):
        done = False
        col = 0

        # Move actions
        if action < 4:
            # Move the agent
            new_pos = self._get_new_position(action)

            # Check for collisions
            col = self._is_collision(new_pos)
            if col == 0:
                self.agent_pos = new_pos
            else:
                self.collision = col
                done = True

        # Pick action
        elif action == 4:
            if not self.agent_has_object:
                for i, obj_pos in enumerate(self.object_positions):
                    if self._distance(self.agent_pos, obj_pos) <= self.pickup_distance:
                        self.agent_has_object = True
                        self.object_positions[i] = None
                        if self.just_pick:
                            done = True
                        break
        
        # Drop action (will only enter if not self.just_pick as the action space has now the action number 5)
        elif action == 5:
            if self.agent_has_object:
                self.agent_has_object = False
                done = True

                if self._is_in_area(self.agent_pos, self.delivery_area):
                    self.agent_has_object = False
                    self.delivery = True
                else:
                    # Drop the object at the agent's position
                    self.object_positions.append(self.agent_pos) 
                    self.agent_has_object = False

        return self._get_obs(), done, {}, col

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos
        for index, object in enumerate(self.object_positions):
            if object is not None:
                obs[2 + 2 * index : 4 + 2 * index] = object
            else:
                obs[2 + 2 * index : 4 + 2 * index] = self.agent_pos
        obs[8] = self.agent_has_object
        obs[9] = self.collision
        obs[10] = self.delivery

        return obs

    def _get_new_position(self, action):
        if action == 0:  # Up
            return (
                self.agent_pos[0],
                min(self.height - self.agent_radius, self.agent_pos[1] + self.agent_velocity)
            )
        elif action == 1:  # Down
            return (
                self.agent_pos[0],
                max(self.agent_radius, self.agent_pos[1] - self.agent_velocity)
            )
        elif action == 2:  # Left
            return (
                max(self.agent_radius, self.agent_pos[0] - self.agent_velocity),
                self.agent_pos[1]
            )
        elif action == 3:  # Right
            return (
                min(self.width - self.agent_radius, self.agent_pos[0] + self.agent_velocity),
                self.agent_pos[1]
            )

    def _is_collision(self, pos):
        # Check for collisions with walls
        if (
            pos[0] <= self.agent_radius
            or pos[0] >= self.width - self.agent_radius
            or pos[1] <= self.agent_radius
            or pos[1] >= self.height - self.agent_radius
        ):
            return 1

        # Check for collisions with shelves
        for shelf in self.shelves:
            if self._is_in_area(pos, shelf, self.agent_radius):
                return 2

        return 0

    def _get_random_empty_position(self):
        while True:
            pos = (
                np.random.uniform(self.agent_radius, self.width - self.agent_radius),
                np.random.uniform(self.agent_radius, self.height - self.agent_radius),
            )
            if not self._is_collision(pos):
                return pos

    def _get_random_position_on_shelf(self, shelf):
        aux = np.random.uniform(0, 1)
        if aux < 0.5:
            x = shelf[0] + 0.25 * shelf[2]
        else:
            x = shelf[0] + 0.75 * shelf[2]
        alpha = 0.5
        y = np.random.uniform(shelf[1] + alpha, shelf[1] + shelf[3] - alpha)
        return (x, y)

    @staticmethod
    def _distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    @staticmethod
    def _is_in_area(pos, area, margin=0):
        return area[0] - margin <= pos[0] <= area[0] + area[2] + margin and area[1] - margin <= pos[1] <= area[1] + area[3] + margin

    def render(self, mode="human", debug=False):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect("equal")

        # Draw shelves
        for shelf in self.shelves:
            self.ax.add_patch(
                Rectangle(
                    shelf[:2],
                    shelf[2],
                    shelf[3],
                    fill=False,
                    edgecolor="brown"
                )
            )

        # Draw delivery area
        self.ax.add_patch(
            Rectangle(
                self.delivery_area[:2],
                self.delivery_area[2],
                self.delivery_area[3],
                fill=True,
                facecolor="lightgreen",
                edgecolor="green", 
                alpha=0.5
            )
        )

        # Draw objects
        for obj_pos in self.object_positions:
            if obj_pos is not None:
                self.ax.add_patch(
                    Circle(
                        obj_pos,
                        radius=0.2,
                        fill=True,
                        facecolor="blue"
                    )
                )

        # Draw agent
        agent_color = "red" if self.agent_has_object else "orange"
        self.ax.add_patch(
                Circle(
                    self.agent_pos,
                    radius=self.agent_radius,
                    fill=True,
                    facecolor=agent_color
                )
            )

        if debug:
            self._render_debug_tiles()

        plt.title(f"Warehouse Environment (Pick: {self.just_pick}, Random: {self.random_objects})")
        plt.draw()
        # save the figure
        # plt.savefig('warehouse.png')
        plt.pause(0.1)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Example usage
if __name__ == "__main__":

    # Select the env mode
    env_mode = 'fixed_drop'  # 'fixed_drop', 'random_drop', 'random_pick'
    if env_mode == 'fixed_pick':
        env = WarehouseEnv(just_pick=True, random_objects=False)
    elif env_mode == 'fixed_drop':
        env = WarehouseEnv(just_pick=False, random_objects=False)
    elif env_mode == 'random_drop':
        env = WarehouseEnv(just_pick=False, random_objects=True)
    elif env_mode == 'random_pick':
        env = WarehouseEnv(just_pick=True, random_objects=True)
    else:
        raise ValueError(f'The env mode: {env_mode} is not available!')
    

    obs = env.reset()
    done = False

    for _ in range(100):  # Run for 100 steps
        action = env.action_space.sample()  # Your agent would make a decision here
        obs, done, _, _ = env.step(action)
        print(f"Action: {action}; Observation: {obs}; done? {done}")
        env.render()
        if done:
            obs = env.reset()

    env.close()
