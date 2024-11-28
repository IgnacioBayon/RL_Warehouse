import numpy as np
from tiles3 import IHT, tiles

class FeedbackConstruction:
    def __init__(self, dims, n_tiles, n_tilings, target_area, rewards = None):
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

        if rewards:
            self.step_reward_obj_1 = rewards["step_reward_obj_1"]
            self.step_reward_obj_2 = rewards["step_reward_obj_2"]
            self.positive_pickup_reward = rewards["positive_pickup_reward"]
            self.collision_reward_1 = rewards["collision_reward_1"]
            self.collision_reward_2 = rewards["collision_reward_2"]
            self.target_reached_reward = rewards["target_reached_reward"]
        else:
            self.step_reward_obj_1 = -0.5
            self.step_reward_obj_2 = -2
            self.positive_pickup_reward = 5000
            self.collision_reward_1 = -100
            self.collision_reward_2 = -5001
            self.target_reached_reward = 10000

        self.flanco = False

        
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
        object_positions = obs[2:8]
        agent_has_object = obs[8]
        collision = obs[9]
        target_area = obs[10]
        
        # Normalize agent position
        norm_x = agent_pos[0] / self.scale_width
        norm_y = agent_pos[1] / self.scale_height
        
        # Get active tiles
        active_tiles = self._get_active_tiles(norm_x, norm_y)

        # Añade aquí tu código para devolver la observación procesada
        observacion = np.concatenate([active_tiles, [agent_has_object, collision, target_area]])
        ##############################

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
    
    def _distance(self, obs):
        """
        Calculate the Euclidean distance between the agent and objects.
        Args:
            pos_agent (tuple): The position of the agent.
            pos_objects (tuple): The position of the objects.
        Returns:
            float: The Euclidean distance between the nearest object and the agent.
        """
        ##############################

        # Añade aquí tu código para calcular la distancia euclidiana
        pos_agent = obs[:2]
        pos_object_1 = obs[2:4]
        pos_object_2 = obs[4:6]
        pos_object_3 = obs[6:8]
        distances = []
        for pos_object in [pos_object_1, pos_object_2, pos_object_3]:
            if pos_object is not None:
                distances.append(np.sqrt((pos_agent[0] - pos_object[0])**2 + (pos_agent[1] - pos_object[1])**2))
        if len(distances) > 0:
            return min(distances)

    def calculate_reward(self, obs):
       
        # Añade aquí tu código para calcular la recompensa
        reward  = 0
 
        agent_pos = obs[:2]
        object_positions = obs[2:8]
        agent_has_object = obs[8]
        collision = obs[9]
        target_area = obs[10]

        # comprobacion de si el agente se encuentra en la zona de entrega
        
        if agent_has_object: 
            if self.flanco == False:
                reward = self.positive_pickup_reward
                self.flanco = True
            else:
                if collision:
                    reward = self.collision_reward_2
                    self.flanco = False
                elif target_area:
                    reward = self.target_reached_reward
                    self.flanco = False
                else: 
                    reward = self.step_reward_obj_2

        else:
            if collision: 
                reward = self.collision_reward_1
            else: 
                reward = self.step_reward_obj_1
            
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
        (warehouse_width, 
         warehouse_height), 
         (n_tiles_width, n_tiles_height), 
         n_tilings, 
        target_area)


