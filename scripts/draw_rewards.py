import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def draw_barplot_results(data, title, save_path, column):
    """
    Draws a grouped bar plot for given data and saves it to the specified path.
    
    Args:
        data (pd.DataFrame): Input data where each row represents a category, 
                             and columns include 'collision_wall', 'collision_obj', 
                             and 'num_objective'.
        title (str): The title of the bar plot.
        save_path (str): The path where the bar plot image will be saved.
    """
    # Transform data to long format for easier plotting
    data_long = data.melt(id_vars=[column], 
                          value_vars=["collision_wall", "collision_obj", "num_objective"],
                          var_name="Type", value_name="Value")
    
    # Reverse the order of the 'col_reward' categories
    # sorted_categories = data[column][::-1]  # Reverse the order
    data_long[column] = pd.Categorical(data_long[column], 
                                             categories=data[column],
                                             ordered=True)
    
    # Set up the color palette
    color_palette = {
        "collision_wall": "red",
        "collision_obj": "orange",
        "num_objective": "green"
    }
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data_long, 
        x=column, 
        y="Value", 
        hue="Type", 
        palette=color_palette, 
        edgecolor="black"  # Add sharp edges to bars
    )
    
    # Add gridlines for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add title and labels
    plt.title(title, fontsize=18, weight='bold')
    plt.xlabel(column, fontsize=14, weight='bold')
    plt.ylabel("Reached", fontsize=14, weight='bold')
    
    # Customize legend
    plt.legend(title="Type", title_fontsize=14, fontsize=12, loc='upper right')
    
    # Tight layout for cleaner appearance
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_dual_barplot(data, title, save_path):
    """
    Draws a grouped bar plot with two y-axes for given data and saves it to the specified path.
    
    Args:
        data (pd.DataFrame): Input data where each row represents a category, 
                             and columns include 'avg_steps' and 'avg_reward'.
        title (str): The title of the bar plot.
        save_path (str): The path where the bar plot image will be saved.
    """
    # Transform data to long format for easier plotting
    data_long = data.melt(id_vars=["col_reward"], 
                          value_vars=["avg_steps", "avg_reward"],
                          var_name="Type", value_name="Value")
    
    # Reverse the order of the 'col_reward' categories
    sorted_categories = data["col_reward"][::-1]  # Reverse the order
    data_long["col_reward"] = pd.Categorical(data_long["col_reward"], 
                                             categories=sorted_categories, 
                                             ordered=True)
    
    # Set up the color palette
    color_palette = {
        "avg_steps": "orange",
        "avg_reward": "blue"
    }
    
    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot avg_steps on the first y-axis (left)
    sns.barplot(
        data=data_long[data_long["Type"] == "avg_steps"], 
        x="col_reward", 
        y="Value", 
        ax=ax1, 
        palette={"avg_steps": "orange"}, 
        edgecolor="black"
    )
    ax1.set_ylabel("Average Steps", fontsize=14, weight='bold', color="orange")
    ax1.tick_params(axis="y", labelcolor="orange")
    ax1.set_xlabel("col_reward", fontsize=14, weight='bold')
    ax1.set_xticklabels(sorted_categories, rotation=90, ha="center", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Create the second y-axis (right) for avg_reward
    ax2 = ax1.twinx()
    sns.barplot(
        data=data_long[data_long["Type"] == "avg_reward"], 
        x="col_reward", 
        y="Value", 
        ax=ax2, 
        palette={"avg_reward": "blue"}, 
        edgecolor="black"
    )
    ax2.set_ylabel("Average Reward", fontsize=14, weight='bold', color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Add title and customize layout
    plt.title(title, fontsize=18, weight='bold')

    # Customize legends
    ax1.legend(["Average Steps"], loc='upper left', fontsize=12, bbox_to_anchor=(0, 1.1))
    ax2.legend(["Average Reward"], loc='upper right', fontsize=12, bbox_to_anchor=(1, 1.1))

    # Tight layout for cleaner appearance
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()





    


if __name__ == "__main__":
    # Example data
    data_col_tr = pd.read_csv("./sensibility/collision_reward_1_tr.csv")
    data_pickup_tr = pd.read_csv("./sensibility/pickup_reward_tr.csv")

    data_col_eval = pd.read_csv("./sensibility/collision_reward_1_eval.csv")
    data_pickup_eval = pd.read_csv("./sensibility/pickup_reward_eval.csv")

    # draw_barplot_results(data_col_tr, "Collision Reward 1", "./sensibility/collision_reward_1_tr_results.png")
    # draw_dual_barplot(data_col_tr, "Collision Reward 1", "./sensibility/collision_reward_1_tr_steps_reward.png")


    draw_barplot_results(data_col_tr, "Collision Reward", "./sensibility/collision_reward_1_tr_results.png", "col_reward")
    draw_barplot_results(data_pickup_tr, "Pickup Reward", "./sensibility/pickup_reward_tr_results.png", "pickup_reward")

    draw_barplot_results(data_col_eval, "Collision Reward", "./sensibility/collision_reward_1_eval_results.png", "col_reward")
    draw_barplot_results(data_pickup_eval, "Pickup Reward", "./sensibility/pickup_reward_eval_results.png", "pickup_reward")

