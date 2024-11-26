# Maze Solver

This is a Python-based application that can solve mazes using various search algorithms, including Breadth-First Search (BFS), Depth-First Search (DFS), A* Search, Iterative Deepening A* (IDA*), Iterative Deepening Search (IDS), and Recursive Best-First Search (RBFS). The application also includes a visualization component to display the solved maze.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Selecting a Search Algorithm](#selecting-a-search-algorithm)
  - [Viewing the Maze Visualization](#viewing-the-maze-visualization)
- [Algorithms](#algorithms)
  - [Breadth-First Search (BFS)](#breadth-first-search-bfs)
  - [Depth-First Search (DFS)](#depth-first-search-dfs)
  - [A* Search](#a-star-search)
  - [Iterative Deepening A* (IDA*)](#iterative-deepening-a-ida)
  - [Iterative Deepening Search (IDS)](#iterative-deepening-search-ids)
  - [Recursive Best-First Search (RBFS)](#recursive-best-first-search-rbfs)
- [Code Structure](#code-structure)
  - [MazeSolver Class](#mazesolver-class)
  - [MazeVisualizer Class](#mazevisualizer-class)
  - [main Function](#main-function)
- [Contributing](#contributing)
- [License](#license)

## Installation
To use the Maze Solver application, you'll need to have Python 3 and the following dependencies installed:

- `tkinter`
- `matplotlib`
- `numpy`
- `heapq`
- `csv`
- `time`

You can install these dependencies using pip:

```
pip install tkinter matplotlib numpy
```

## Usage

### Running the Application
To run the Maze Solver application, execute the `main.py` file:

```
python main.py
```

This will launch the graphical user interface (GUI) where you can interact with the application.

### Selecting a Search Algorithm
The GUI provides buttons for each of the available search algorithms:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search
- Iterative Deepening A* (IDA*)
- Iterative Deepening Search (IDS)
- Recursive Best-First Search (RBFS)

Click on the button for the algorithm you want to use to solve the maze.

### Viewing the Maze Visualization
After selecting a search algorithm and providing a maze file, the application will display the solved maze in a separate window. The solution path will be highlighted in red.

## Algorithms

### Breadth-First Search (BFS)
Breadth-First Search is a graph traversal algorithm that explores all the neighboring nodes at the present depth before moving on to the nodes at the next depth level. In the provided code, the `bfs_search` method implements this algorithm:

```python
def bfs_search(self):
    queue = [(self.start, [self.start])]
    visited = set([self.start])
    expanded_nodes = 0

    while queue:
        current, path = queue.pop(0)
        if current == self.goal:
            return {'path': path, 'path_cost': len(path) - 1, 'expanded_nodes': expanded_nodes}

        for dx, dy in self.directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in visited and self.is_valid_move(*neighbor):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                expanded_nodes += 1

    return None
```

1. A `queue` is used to keep track of the nodes to be explored.
2. A `visited` set is used to keep track of the nodes that have been visited.
3. The `expanded_nodes` variable is used to count the number of nodes explored.
4. Each neighboring node is checked for validity and added to the queue if it hasn't been visited before.

### Depth-First Search (DFS)
Depth-First Search is a graph traversal algorithm that explores as far as possible along each branch before backtracking. In the provided code, the `dfs_search` method implements this algorithm:

```python
def dfs_search(self):
    start_time = time.time()
    stack = [self.start]
    visited = set([self.start])
    came_from = {}
    expanded_nodes = 0
    unsuccessful_attempts = 0

    while stack:
        current = stack.pop()
        expanded_nodes += 1

        if current == self.goal:
            path = self.reconstruct_path(came_from, current)
            return {
                'path': path,
                'path_cost': len(path) - 1,
                'expanded_nodes': expanded_nodes,
                'time': time.time() - start_time,
                'unsuccessful_attempts': unsuccessful_attempts
            }

        for dx, dy in self.directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if self.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in visited:
                stack.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current
            else:
                unsuccessful_attempts += 1
    
    return None
```

1. A `stack` is used to keep track of the nodes to be explored.
2. A `visited` set is used to keep track of the nodes that have been visited.
3. A `came_from` dictionary is used to reconstruct the solution path.
4. The `expanded_nodes` variable is used to count the number of nodes explored, and the `unsuccessful_attempts` variable tracks the number of invalid moves.
5. Each neighboring node is checked for validity and added to the stack if it hasn't been visited before.

The remaining algorithm implementations (A* Search, Iterative Deepening A* (IDA*), Iterative Deepening Search (IDS), and Recursive Best-First Search (RBFS)) are also provided in the code, and you can find their detailed explanations in the README file.

## Code Structure

### MazeSolver Class
The `MazeSolver` class is responsible for loading the maze from a CSV file, implementing the various search algorithms, and providing utility functions for working with the maze data.

### MazeVisualizer Class
The `MazeVisualizer` class is responsible for rendering the maze and the solution path using the `matplotlib` library.

### main Function
The `main` function is the entry point of the application. It sets up the Tkinter-based GUI, allows the user to select a search algorithm and a maze file, and calls the appropriate functions to solve the maze and display the visualization.

## Contributing
If you find any issues or have suggestions for improvements, feel free to create a new issue or submit a pull request on the project's GitHub repository.

## License
This project is licensed under the [MIT License](LICENSE).
