import numpy as np
from representacion_sens import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from almacen_all import WarehouseEnv
import time
import pandas as pd

class SarsaAgent:
    """
    SarsaAgent is an implementation of the SARSA(0) algorithm for reinforcement learning.
    Attributes:
        env (gym.Env): The environment in which the agent operates.
        feedback (object): An object that processes observations from the environment.
        learning_rate (float): The learning rate for updating the weights.
        discount_factor (float): The discount factor for future rewards.
        epsilon (float): The probability of choosing a random action (exploration rate).
        num_actions (int): The number of possible actions in the environment.
        feature_size (int): The size of the feature vector for each state.
        weights (list of np.ndarray): The weights for each action.
    Methods:
        __init__(env, gateway, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
            Initializes the SarsaAgent with the given parameters.
        get_action(state, epsilon=None):
            Returns an action based on the epsilon-greedy policy.
        get_q_values(state):
            Computes the Q-values for all actions given the current state.
        update(state, action, reward, next_state, next_action):
            Updates the weights based on the SARSA update rule.
        train(num_episodes):
            Trains the agent for a specified number of episodes.
        evaluate(num_episodes):
            Evaluates the agent's performance over a specified number of episodes.
    """
    def __init__(self, env: WarehouseEnv, feedback: FeedbackConstruction, learning_rate=0.01, discount_factor=0.99, epsilon=0.1):
        # Mejor no toques estas líneas
        self.env = env
        self.feedback = feedback
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = env.action_space.n
        self.feature_size = feedback.iht.size
        ##################################
        
        # Si vas a añadir más variables (features) además del tile coding, resérvales espacio aquí.
        ##############################

        # Te damos los pesos inicializados a cero. Pero esto es arbitrario. Lo puedes cambiar si quieres.
        self.weights = [np.zeros(self.feature_size) for _ in range(self.num_actions)]
        
        # Tendrás que usar estrategias para monitorizar el aprendizaje del agente.
        # Añade aquí los atributos que necesites para hacerlo.

        ##############################

    def get_action(self, state, epsilon=None):
        """
        Selects an action based on the epsilon-greedy policy.
        Parameters:
        state (object): The current state of the environment.
        epsilon (float, optional): The probability of selecting a random action. 
                                   If None, the default epsilon value is used.
        Returns:
        int: The selected action.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            # Random action
            return self.env.action_space.sample()
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def get_q_values(self, state):
        """
        Computes the Q-values of all actions for a given state.

        Parameters:
        state (object): The current state for which Q-values need to be computed.

        Returns:
        np.ndarray: A numpy array of Q-values for each action in the given state.
        """
        
        features = self.feedback.process_observation(state)
        q_values = np.zeros(self.num_actions)
        indx = [int(t) for t in features[:-2]]
        # Calcula los valores de cada acción para el estado dado como argumento (aproximación
        # lineal). Añade aquí tu código
        for i in range(self.num_actions):
           weights = self.weights[i][indx]
           q_values[i] = np.sum(weights)
        
        return q_values
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update the weights for the given state-action pair using the SARSA(0) algorithm.
        Parameters:
        state (object): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (object): The state resulting from taking the action.
        next_action (int): The action to be taken in the next state.
        Returns:
        None
        """
        qs_current = self.get_q_values(state)       
        q_current = qs_current[action]
        # td_error
        if done:
            td_error = reward - q_current
        else:
            qs_next = self.get_q_values(next_state)

            q_next = qs_next[next_action]            
            td_error = reward + self.discount_factor * q_next - q_current

        # TODO actualizar los pesos del agente
        features = self.feedback.process_observation(state)[:-2]
        indx = [int(t) for t in features]

        for i in indx:
            self.weights[action][i] += self.learning_rate * td_error

    def train(self, num_episodes):
        """
        Train the agent using the SARSA(0) algorithm.
        Parameters:
        num_episodes (int): The number of episodes to train the agent.
        The method runs the training loop for the specified number of episodes.
        In each episode, the agent interacts with the environment, selects actions
        based on the current policy, and updates the policy using the SARSA(0) update rule.
        The total reward for each episode is printed every 100 episodes.
        Returns:
        None
        """

        episodes = 0
        collision_wall = 0
        collision_obj = 0
        num_objective = 0
        avg_reward = 0
        avg_steps = 0
        time_ini = time.time()

        epsilon = self.epsilon
        # Juega con estos tres hiperparámetros:
        # entre 0 y 1. 
        decay_start = .6
        # control del decrecimiento (exponencial) de epsilon
        decay_rate = 0.999
        # valor mínimo de epsilon
        min_epsilon = .000
        
        ####################################
        for episode in range(num_episodes):
            # Set-up del episodio
            state = self.env.reset()
            self.feedback.flanco = False
            # Decrecimiento exponencial de epsilon hasta valor mínimo desde comienzo marcado
            if episode >= num_episodes*decay_start:
                epsilon *= decay_rate
                epsilon = np.max([min_epsilon, epsilon])
            # Primera acción
            action = self.get_action(state, epsilon)            
            n_steps = 0
            # Generación del episodio
            total_undiscounted_return = 0
            while True:                                        
                next_state, done, _, col = self.env.step(action)
                reward = self.feedback.calculate_reward(next_state)      
                total_undiscounted_return += reward          
                next_action = self.get_action(next_state, epsilon)
                self.update(state, action, reward, next_state, next_action, done)    
                state = next_state
                action = next_action                
                n_steps += 1
                if done:

                    if col == 1:
                        collision_wall += 1

                    if col == 2:
                        collision_obj += 1

                    if next_state[8]:
                        num_objective += 1

                    break
                # Esto se añade por si puntualmente hubiera alguna configuración
                # de qs que hiciera al agente oscilar demasiado entre dos estados
                # Estrictamente hablando, no es necesario, pero lo hace más 
                # eficiente.
                if n_steps >= 2000:
                    break
            episodes += 1
            avg_reward += total_undiscounted_return
            avg_steps += n_steps

        return episodes, collision_wall, collision_obj, num_objective, avg_reward, avg_steps, time.time() - time_ini

    
    def evaluate(self, num_episodes):
        """
        Evaluate the agent's performance over a specified number of episodes.
        Parameters:
        num_episodes (int): The number of episodes to run the evaluation.
        Returns:
        float: The average reward obtained over the specified number of episodes.
        This method runs the agent in the environment for a given number of episodes
        using a greedy policy (epsilon=0). It collects the total reward for each episode
        and computes the average reward over all episodes. The average reward is printed
        and returned.
        Note:
        - The environment is reset at the beginning of each episode.
        - The agent's action is determined by the `get_action` method with epsilon set to 0.
        """
        steps = 0
        num_objective = 0
        collision_wall = 0
        collision_obj = 0
        episodes = 0

        for _ in range(num_episodes):
            romper = 0
            state = self.env.reset()
            total_undiscounted_return = 0
            done = False

            while not done:
                action = self.get_action(state, epsilon=0.02)  # Greedy policy
                next_state, done, _, col = self.env.step(action)
                reward = self.feedback.calculate_reward(next_state)
                if done:
                    
                    if col == 1:
                        collision_wall += 1

                    if col == 2:
                        collision_obj += 1

                    if next_state[8]:
                        num_objective += 1

                # self.env.render()
                state = next_state
                total_undiscounted_return += reward
                steps += 1
                romper += 1
                if romper > 2000:
                    break

            episodes += 1
 
        return episodes, total_undiscounted_return, steps, collision_wall, collision_obj, num_objective


if __name__ == "__main__":
    # instanciamos entorno, representación y agente
    # No tocar
    env = WarehouseEnv(just_pick=True, random_objects=False)
    warehouse_width = 10.0
    warehouse_height = 10.0
    ################
    # diseñar los tiles
    n_tiles_width = 15
    n_tiles_height = 15
    n_tilings = 10
    learning_rate = 0.005
    discount_factor = 0.99995
    epsilon = 0.6
    tr_episodes = 10000
    eval_episodes = 500
    
    target_area = (2.5, 8, 1.0, 2.0)

    # We will analyse the following rewards in terms of proportionalities between "step_reward_obj_1"

    reward_obj_1 = -0.5
    base_collision_reward = -100
    base_pickup_reward = 5000

    prop_r1_collision = [10, 50, 100, 200, 300, 400]
    prop_r1_pickup = [1000, 5000, 10000, 20000, 30000, 40000]

    rewards = {
        "step_reward_obj_1": -0.5,
        "step_reward_obj_2": -2,
        "positive_pickup_reward": 5000,
        "collision_reward_1": -100,
        "collision_reward_2": -5001,
        "target_reached_reward": 10000
    }

    df_col_tr = pd.DataFrame(columns=["col_reward","episodes", "collision_wall", "collision_obj", "num_objective", "avg_reward", "avg_steps", "time"])

    df_col_eval = pd.DataFrame(columns=["col_reward", "episodes", "total_undiscounted_return", "steps", "collision_wall", "collision_obj", "num_objective"])
    # We will train many agents with different parameters mainatining the rest of the parameters constant

    for reward in prop_r1_collision:

        print('Training agent with collision reward: ', reward * reward_obj_1)
        rewards["collision_reward_1"] = reward * reward_obj_1
        
        feedback = FeedbackConstruction(
            (warehouse_width, warehouse_height),
            (n_tiles_width, n_tiles_height),
            n_tilings,
            target_area, rewards
        )

        agent = SarsaAgent(
            env,
            feedback,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )

        episodes, collision_wall, collision_obj, num_objective, avg_reward, avg_steps, time_tr = agent.train(tr_episodes)


        print('Parameters: ')
        print('N-tilings: ', n_tilings)
        print('(W, H): ', (n_tiles_width, n_tiles_height))
        print('LR: ', learning_rate)
        print('DF: ', discount_factor)
        print('EPS: ', epsilon)

        # Add to the dataframe
        df_col_tr = df_col_tr._append({
            "col_reward": reward * reward_obj_1,
            "episodes": episodes,
            "collision_wall": collision_wall,
            "collision_obj": collision_obj,
            "num_objective": num_objective,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "time": time_tr
        }, ignore_index=True)

        # Evaluate the agent
        episodes, total_undiscounted_return, steps, collision_wall, collision_obj, num_objective = agent.evaluate(eval_episodes)
        df_col_eval = df_col_eval._append({
            "col_reward": reward * reward_obj_1,
            "episodes": episodes,
            "total_undiscounted_return": total_undiscounted_return,
            "steps": steps,
            "collision_wall": collision_wall,
            "collision_obj": collision_obj,
            "num_objective": num_objective
        }, ignore_index=True)


    df_col_tr.to_csv("./sensibility/collision_reward_1_tr.csv")
    df_col_eval.to_csv("./sensibility/collision_reward_1_eval.csv")

    df_pickup_tr = pd.DataFrame(columns=["pickup_reward","episodes", "collision_wall", "collision_obj", "num_objective", "avg_reward", "avg_steps","time"])

    df_pickup_eval = pd.DataFrame(columns=["pickup_reward","episodes", "total_undiscounted_return", "steps", "collision_wall", "collision_obj", "num_objective"])

    rewards["collision_reward_1"] = base_collision_reward
    for reward in prop_r1_pickup:
        print('Training agent with pickup reward: ', reward)
        rewards["positive_pickup_reward"] = reward * reward_obj_1

        feedback = FeedbackConstruction(
            (warehouse_width, warehouse_height),
            (n_tiles_width, n_tiles_height),
            n_tilings,
            target_area, rewards
        )

        agent = SarsaAgent(
            env,
            feedback,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )

        episodes, collision_wall, collision_obj, num_objective, avg_reward, avg_steps, time_tr = agent.train(tr_episodes)


        print('Parameters: ')
        print('N-tilings: ', n_tilings)
        print('(W, H): ', (n_tiles_width, n_tiles_height))
        print('LR: ', learning_rate)
        print('DF: ', discount_factor)
        print('EPS: ', epsilon)

        # Add to the dataframe
        df_pickup_tr = df_pickup_tr._append({
            "pickup_reward": reward*reward_obj_1,
            "episodes": episodes,
            "collision_wall": collision_wall,
            "collision_obj": collision_obj,
            "num_objective": num_objective,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "time": time_tr
        }, ignore_index=True)

        # Evaluate the agent
        episodes, total_undiscounted_return, steps, collision_wall, collision_obj, num_objective = agent.evaluate(eval_episodes)

        df_pickup_eval = df_pickup_eval._append({
            "pickup_reward": reward*reward_obj_1,
            "episodes": episodes,
            "total_undiscounted_return": total_undiscounted_return,
            "steps": steps,
            "collision_wall": collision_wall,
            "collision_obj": collision_obj,
            "num_objective": num_objective
        }, ignore_index=True)

    df_pickup_tr.to_csv("./sensibility/pickup_reward_tr.csv")
    df_pickup_eval.to_csv("./sensibility/pickup_reward_eval.csv")





        

    
    
    # # save the agent object into memory    
    # with open('Anacleto_Agente_Secreto_VERSION_MEJORADA', 'wb') as f:
    #     pickle.dump(agent, f)

    

    # # load the agent object from memory
    # with open('Anacleto_Agente_Secreto_VERSION_MEJORADA.pkl', 'rb') as f:
    #     agent = pickle.load(f)

    # # Evaluate the agent


    # agent.evaluate(num_episodes=10000)