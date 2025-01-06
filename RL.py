# 随机迷宫生成 + 增强学习找最优路径

import numpy as np
import random
import matplotlib.pyplot as plt

# Define the size of the maze
MAZE_SIZE = 30

# Define the action space
ACTIONS = ['up', 'down', 'left', 'right']

# Define rewards
REWARD_GOAL = 10
REWARD_WALL = -5
REWARD_STEP = -1

# Define Q-learning parameters
LEARNING_RATE = 0.03
DISCOUNT_FACTOR = 0.9
EPISODES = 30000
EPSILON = 0.1
EPSILON_DECAY = 0.995  # Decay rate for epsilon

LOG_INTERVAL = 100

# Function to generate a random maze
def generate_maze(size):
    maze = np.zeros((size, size))
    # Randomly place obstacles (20% probability)
    for i in range(size):
        for j in range(size):
            if random.random() < 0.2:  # 20% chance to place an obstacle
                maze[i][j] = 1
    # Set the start and goal positions
    maze[0][0] = 0  # Start position
    maze[size-1][size-1] = 2  # Goal position
    return maze

# Function to print the maze
def print_maze(maze):
    for row in maze:
        print(" ".join(str(cell) for cell in row))

# Function to visualize the maze
def visualize_maze(maze, title="Generated Maze"):
    plt.figure()  # Create a new figure window
    plt.imshow(maze, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

def visualize_maze_non_block(maze, title="Generated Maze"):
    plt.figure()  # Create a new figure window
    plt.imshow(maze, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)  # 不阻塞程序
    plt.pause(1)  # 暂停1秒，确保图形窗口有时间更新

# Function to get the next state based on the current state and action
def get_next_state(state, action):
    i, j = state
    if action == 'up':
        i = max(i - 1, 0)
    elif action == 'down':
        i = min(i + 1, MAZE_SIZE - 1)
    elif action == 'left':
        j = max(j - 1, 0)
    elif action == 'right':
        j = min(j + 1, MAZE_SIZE - 1)
    return (i, j)

# Function to get the reward for moving to the next state
def get_reward(maze, next_state):
    i, j = next_state
    if maze[i][j] == 1:  # Hit a wall
        return REWARD_WALL
    elif maze[i][j] == 2:  # Reached the goal
        return REWARD_GOAL
    else:
        return REWARD_STEP  # Default step reward

# Q-learning algorithm
def q_learning(maze):
    print("Start Q-learning")
    # Initialize the Q-table with zeros
    Q = np.zeros((MAZE_SIZE, MAZE_SIZE, len(ACTIONS)))
    print("Q table size = ", Q.shape)
    epsilon = EPSILON
    
    for episode in range(EPISODES):
        state = (0, 0)  # Start from the initial state
        if (episode + 1) % LOG_INTERVAL == 0:  # Check if the episode is a multiple of 10
            print(f"\n--- Episode {episode + 1} ---")
        step = 0
        while True:
            step += 1
            
            # Choose an action (epsilon-greedy strategy)
            if random.random() < epsilon:
                action = random.choice(ACTIONS)  # Explore: choose a random action
            else:
                action = ACTIONS[np.argmax(Q[state[0], state[1]])]  # Exploit: choose the best action
            
            # Execute the action and observe the next state and reward
            next_state = get_next_state(state, action)
            reward = get_reward(maze, next_state)
            
            # Update the Q-value using the Bellman equation
            old_q_value = Q[state[0], state[1], ACTIONS.index(action)]
            Q[state[0], state[1], ACTIONS.index(action)] += LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * np.max(Q[next_state[0], next_state[1]]) - old_q_value
            )
            
            # Check if the goal is reached
            if maze[next_state[0], next_state[1]] == 2:
                if (episode + 1) % LOG_INTERVAL == 0:  # Only print if the episode is a multiple of 10
                    print(f"Goal reached in {step} steps! Episode finished.")
                break
            
            # Move to the next state
            state = next_state
        
        # Decay epsilon
        epsilon *= EPSILON_DECAY
    
    return Q

# Function to test the learned policy
def test_policy(maze, Q):
    state = (0, 0)  # Start from the initial state
    path = [state]
    print("\nTesting the learned policy:")
    step = 0
    while True:
        step += 1
        # Choose the best action based on the Q-table
        action = ACTIONS[np.argmax(Q[state[0], state[1]])]
        next_state = get_next_state(state, action)
        path.append(next_state)
        
        # Check if the goal is reached
        if maze[next_state[0], next_state[1]] == 2:
            print(f"Goal reached in {step} steps during policy testing!")
            break
        
        # Move to the next state
        state = next_state
    return path

# Main function
def main():
    # Generate the maze
    maze = generate_maze(MAZE_SIZE)
    print("Generated Maze:")
    print_maze(maze)
    
    # Visualize the generated maze
    visualize_maze_non_block(maze, title="Generated Maze")
    
    # Train the Q-learning agent
    Q = q_learning(maze)
    
    # save Q table
    np.save("q_table.npy", Q)
    print("Q-table saved to 'q_table.npy'")

    # # print Q table
    # print("Q-table after training:")
    # print(Q)

    # Test the learned policy
    path = test_policy(maze, Q)
    print("\nPath found by Q-learning:")
    for p in path:
        print(p)

    # Visualize the path
    maze_with_path = np.copy(maze)
    for p in path:
        maze_with_path[p[0], p[1]] = 3  # Mark the path
    
    visualize_maze(maze_with_path, title="Maze with Q-learning Path")

if __name__ == "__main__":
    main()
