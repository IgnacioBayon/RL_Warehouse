import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

def draw_reward_training_graph(data, title, save_path):
    """
    Draws a reward training graph for the given data.
    
    Parameters:
        data (pd.DataFrame): The input data containing reward columns.
        title (str): Title of the graph.
        save_path (str): Path to save the generated graph.
    """
    plt.figure(figsize=(10, 6))
    
    # Loop through each series in the dataframe
    for col in data.columns:
        if col.startswith("reward_"):
            label = col.split("_")[-1]
            plt.plot(data[col], label=f"{label}")
    
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def draw_steps_training_graph(data, title, save_path):
    """
    Draws a steps training graph for the given data.
    
    Parameters:
        data (pd.DataFrame): The input data containing step columns.
        title (str): Title of the graph.
        save_path (str): Path to save the generated graph.
    """
    plt.figure(figsize=(10, 6))
    
    # Loop through each series in the dataframe
    for col in data.columns:
        if col.startswith("steps_"):
            label = col.split("_")[-1]
            plt.plot(data[col], label=f"{label}")
    
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def draw_success_training_graph(data, title, save_path):
    """
    Draws a success rate training graph for the given data.
    
    Parameters:
        data (pd.DataFrame): The input data containing end columns.
        title (str): Title of the graph.
        save_path (str): Path to save the generated graph.
    """
    plt.figure(figsize=(10, 6))
    
    # Loop through each series in the dataframe
    for col in data.columns:
        if col.startswith("end_"):
            label = col.split("_")[-1]
            # Count only zeros, ignoring other values and NaN
            zero_count = (data[col] == 0).cumsum()
            print(f"{col} - {zero_count}")
            success_rate = zero_count / (data.index + 1)
            plt.plot(success_rate, label=f"Epsilon {label}")

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()



if __name__ == "__main__":

    types = [
        "learning_rate",
        "epsilon",
        "discount_factor",
        "n_tilings",
        "n_wh"
    ]

    # Load the data

    for type in types:
        data = pd.read_csv(f"./train_approach/results_{type}.csv")

        # Draw the graphs
        # draw_reward_training_graph(data, f"{type} Sensibility", f"./train_approach/{type}_reward.png")
        # draw_steps_training_graph(data, f"{type} Sensibility", f"./train_approach/{type}_steps.png")
        draw_success_training_graph(data, f"{type} Sensibility", f"./train_approach/{type}_success.png")




