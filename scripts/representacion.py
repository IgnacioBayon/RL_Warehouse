import numpy as np
from tiles3 import IHT, tiles


agent = 1


class FeedbackConstruction:
    def __init__(self, dims, n_tiles, n_tilings, target_area):
        # No tocar estas líneas
        self.width = dims[0]
        self.height = dims[1]
        self.scale_width = dims[0] / n_tiles[0]
        self.scale_height = dims[1] / n_tiles[1]
        self.target_area = target_area        
        self.num_tilings = n_tilings
        self.max_size = n_tiles[0] * n_tiles[1] * self.num_tilings + 2000
        self.iht = IHT(self.max_size)
        ##############################
        # Si quieres añadir más atributos, añádelos a partir de aquí
        if agent == 1:
            self.step_reward = -1
            self.collision_reward = -10
            self.target_reached_reward = 20
            ##############################
            self.wrong_pick = -1
            self.pick_object = 5
        if agent == 2 or agent == 3:
            self.step_reward = -1
            self.collision_reward = -10
            self.target_reached_reward = 20
            ##############################
            self.wrong_pick = -1
            self.wrong_drop = -1
            self.pick_object = 5
            self.drop_object = 5
        self.had_object = False

    def process_observation(self, obs):
        """
        Processes the observation from the environment and extracts relevant features.
        Args:
            obs (list or np.ndarray): The observation from the environment. It is expected to be a list or array
                                      where the first two elements are the agent's position, the next six elements
                                      are the positions of three objects (each represented by two elements), the 
                                      eighth element indicates if the agent has an object, and the ninth element 
                                      indicates if there is a collision.
        Returns:
            np.ndarray: A concatenated array of active tiles, user-defined variables, agent's object possession status,
                        and collision status.
        """
        agent_pos = obs[:2]
        collision = obs[2]
        target_area = obs[3]
        
        # Normalize agent position
        norm_x = agent_pos[0] / self.scale_width
        norm_y = agent_pos[1] / self.scale_height
        
        # Get active tiles
        active_tiles = self._get_active_tiles(norm_x, norm_y)

        # Añade aquí tu código para devolver la observación procesada
        observacion = np.concatenate([active_tiles, [collision], [target_area]])

        return observacion

    def _get_active_tiles(self, norm_x, norm_y):
        """
        Calculate the active tiles for given normalized x and y coordinates.
        This method computes the active tiles based on the normalized coordinates
        (norm_x, norm_y) and the number of tilings. It applies specific offsets
        for each tiling to determine the active tiles.
        Args:
            norm_x (float): Normalized x-coordinate.
            norm_y (float): Normalized y-coordinate.
        Returns:
            list: A list of active tile indices.
        """
        # Añade aquí tu código para calcular los tiles activos                         
        offset_factor_x = 1/self.num_tilings * 3
        offset_factor_y = 1/self.num_tilings * 1
        active_tiles = []
        for i in range(self.num_tilings):
            offset_x = offset_factor_x * i
            offset_y = offset_factor_y * i
            tile_temp = tiles(self.iht, 1, 
                    [norm_x - offset_x, 
                    norm_y - offset_y],
                    ints=[i])
            active_tiles.append(tile_temp[0])
                
        return active_tiles
        ##############################

    def calculate_reward(self, obs):
       
        # Añade aquí tu código para calcular la recompensa
        reward  = 0
 
        agent_pos = obs[:2]
        pos_obj1 = obs[2:4]
        pos_obj2 = obs[4:6]
        pos_obj3 = obs[6:8]
        has_object = obs[8]
        collision = obs[9]
        # True if drop in target area. False otherwise
        delivery = obs[10]

        if collision:
            reward = self.collision_reward
        elif delivery:
            if has_object:
                reward = self.drop_object
            else:
                reward = self.wrong_drop
        elif not delivery:
            if self.had_object and not has_object:
                reward = self.wrong_drop
                self.pick_object = 5
            elif has_object and not self.had_object:
                reward = self.pick_object
                self.pick_object = -1
        else:
            reward = self.step_reward

        self.had_object = has_object

        return reward


    
if __name__ == "__main__":
    # Espacio para pruebas

    # No hay por qué tocar este código
    warehouse_width = 10.0
    warehouse_height = 10.0
    target_area = (2.5, 8, 5.0, 2.0)
    ##############################
    # Libertad total desde aquí
    n_tiles_width = 1
    n_tiles_height = 1
    n_tilings = 4


    realimentacion = FeedbackConstruction(
        (warehouse_width, warehouse_height),
        (n_tiles_width, n_tiles_height),
        n_tilings, 
        target_area
    )


