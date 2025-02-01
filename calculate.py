import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to initialize the 3D grid with initial temperatures
def initialize_grid(dim_x, dim_y, dim_z, initial_temp):
    return np.full((dim_x, dim_y, dim_z), initial_temp)

# Function to apply specific temperature values at user-defined points
def apply_temperature(grid, coordinates, temperature):
    for coord in coordinates:
        x, y, z = coord
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]:
            grid[x, y, z] = temperature
        else:
            print(f"Warning: Coordinate ({x},{y},{z}) is out of bounds!")

# Function to calculate the average temperature of the grid
def calculate_average_temperature(grid):
    return np.mean(grid)

# Function to display the temperature distribution in a 3D plot
def plot_3d_temperature_distribution(grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the grid
    x, y, z = np.indices(grid.shape)

    # Flatten the grid for plotting
    temp_values = grid.flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Plot the temperature as a scatter plot
    scatter = ax.scatter(x_flat, y_flat, z_flat, c=temp_values, cmap='hot')

    # Add color bar
    fig.colorbar(scatter, ax=ax, label="Temperature (°C)")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def main():
    # User inputs for grid size and initial temperature
    dim_x = int(input("Enter the number of points along X axis: "))
    dim_y = int(input("Enter the number of points along Y axis: "))
    dim_z = int(input("Enter the number of points along Z axis: "))
    initial_temp = float(input("Enter the initial temperature for the grid (°C): "))

    # Initialize the grid with the specified temperature
    grid = initialize_grid(dim_x, dim_y, dim_z, initial_temp)

    # User input for specific temperature points
    num_points = int(input("Enter the number of points to apply specific temperatures: "))
    temp_points = []
    
    for i in range(num_points):
        x = int(input(f"Enter x coordinate for point {i+1}: "))
        y = int(input(f"Enter y coordinate for point {i+1}: "))
        z = int(input(f"Enter z coordinate for point {i+1}: "))
        temperature = float(input(f"Enter the temperature for point ({x},{y},{z}): "))
        temp_points.append((x, y, z, temperature))

    # Apply the specific temperature values
    for point in temp_points:
        apply_temperature(grid, [(point[0], point[1], point[2])], point[3])

    # Calculate and print the average temperature
    avg_temp = calculate_average_temperature(grid)
    print(f"Average temperature of the grid: {avg_temp:.2f}°C")

    # Plot the 3D temperature distribution
    plot_3d_temperature_distribution(grid)

if __name__ == "__main__":
    main()
