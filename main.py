import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Listbox
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import heapq

class MazeSolver:
    def __init__(self, maze_file):
        self.maze = self.load_maze(maze_file)
        self.start = (0, 0)
        self.goal = (len(self.maze) - 1, len(self.maze[0]) - 1)
        self.rows = len(self.maze)
        self.cols = len(self.maze[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def load_maze(self, maze_file):
        try:
            with open(maze_file, 'r') as f:
                reader = csv.reader(f)
                maze = []
                for row in reader:
                    maze.append([int(float(cell)) for cell in row])  # convert int to float
                return maze
        except FileNotFoundError:
            raise FileNotFoundError(f"File {maze_file} not found.")
        except ValueError as e:
            raise ValueError(f"Error loading maze: {str(e)}")

    def is_valid_move(self, x, y):
        return (0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] == 1)

    def manhattan_distance(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
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

    def a_star_search(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.manhattan_distance(self.start, self.goal)}

        expanded_nodes = 0

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == self.goal:
                path = self.reconstruct_path(came_from, current)
                return {'path': path, 'path_cost': len(path) - 1, 'expanded_nodes': expanded_nodes}

            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self.is_valid_move(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    expanded_nodes += 1

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
    def rbfs_search(self):
        start_time = time.time()
        self.expanded_nodes = 0
        self.unsuccessful_attempts = 0

        def rbfs_search(self):
            start_time = time.time()
            expanded_nodes = 0
            unsuccessful_attempts = 0

            def rbfs(node, g_score, f_limit):
                nonlocal expanded_nodes, unsuccessful_attempts

                if node == self.goal:
                    return [node], 0

                successors = []
                for dx, dy in self.directions:
                    neighbor = (node[0] + dx, node[1] + dy)
                    if self.is_valid_move(neighbor[0], neighbor[1]):
                        expanded_nodes += 1
                        g_cost = g_score + 1
                        h_cost = self.manhattan_distance(neighbor, self.goal)
                        f_cost = max(g_cost + h_cost, f_limit)
                        successors.append((neighbor, f_cost, g_cost))
                    else:
                        unsuccessful_attempts += 1

                if not successors:
                    return None, float('inf')

                successors.sort(key=lambda x: x[1])

                while True:
                    best_node, best_f, best_g = successors[0]
                    alternative_f = successors[1][1] if len(successors) > 1 else float('inf')

                    if best_f > f_limit:
                        return None, best_f

                    result, best_f = rbfs(best_node, best_g, min(f_limit, alternative_f))

                    successors[0] = (best_node, best_f, best_g)
                    successors.sort(key=lambda x: x[1])

                    if result is not None:
                        return [node] + result, best_f

            result, _ = rbfs(self.start, 0, float('inf'))

            if result:
                return {
                    'path': result,
                    'path_cost': len(result) - 1,
                    'expanded_nodes': expanded_nodes,
                    'time': time.time() - start_time,
                    'unsuccessful_attempts': unsuccessful_attempts
                }

            return None

    def ida_star_search(self):
        start_time = time.time()
        self.expanded_nodes = 0
        self.unsuccessful_attempts = 0

        def search(path, g, bound, visited):
            node = path[-1]
            f = g + self.manhattan_distance(node, self.goal)

            if f > bound:
                return f

            if node == self.goal:
                return True

            self.expanded_nodes += 1
            min_cost = float('inf')

            for dx, dy in self.directions:
                next_node = (node[0] + dx, node[1] + dy)

                if not self.is_valid_move(next_node[0], next_node[1]) or next_node in visited:
                    self.unsuccessful_attempts += 1
                    continue

                if next_node not in path:
                    path.append(next_node)
                    visited.add(next_node)
                    result = search(path, g + 1, bound, visited)

                    if result is True:
                        return True

                    if result < min_cost:
                        min_cost = result

                    path.pop()
                    visited.remove(next_node)

            return min_cost

        bound = self.manhattan_distance(self.start, self.goal)
        path = [self.start]
        visited = {self.start}

        while True:
            result = search(path, 0, bound, visited)
            if result is True:
                return {
                    'path': path,
                    'path_cost': len(path) - 1,
                    'expanded_nodes': self.expanded_nodes,
                    'time': time.time() - start_time,
                    'unsuccessful_attempts': self.unsuccessful_attempts
                }
            if result == float('inf'):
                return None
            bound = result

    def ids_search(self):
        start_time = time.time()
        self.expanded_nodes = 0
        self.unsuccessful_attempts = 0

        def dls(node, depth, max_depth, visited, path):
            if depth > max_depth:
                return None

            if node == self.goal:
                return path

            self.expanded_nodes += 1

            for dx, dy in self.directions:
                next_node = (node[0] + dx, node[1] + dy)

                if not self.is_valid_move(next_node[0], next_node[1]) or next_node in visited:
                    self.unsuccessful_attempts += 1
                    continue

                new_visited = visited | {next_node}
                new_path = path + [next_node]

                result = dls(next_node, depth + 1, max_depth, new_visited, new_path)
                if result is not None:
                    return result

            return None

        # Try increasing depths until solution is found or maze size is reached
        max_possible_depth = self.rows * self.cols

        for depth in range(max_possible_depth):
            visited = {self.start}
            path = dls(self.start, 0, depth, visited, [self.start])

            if path is not None:
                return {
                    'path': path,
                    'path_cost': len(path) - 1,
                    'expanded_nodes': self.expanded_nodes,
                    'time': time.time() - start_time,
                    'unsuccessful_attempts': self.unsuccessful_attempts
                }

        return None
    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))


class MazeVisualizer:
    def __init__(self, maze, path):
        self.maze = np.array(maze)
        self.path = path

    def display_maze(self):
        rows, cols = self.maze.shape
        fig, ax = plt.subplots(figsize=(cols / 2, rows / 2))

        ax.imshow(self.maze, cmap='binary', origin='upper')

        if self.path:
            x, y = zip(*self.path)
            ax.plot(y, x, marker='o', color='red', markersize=5)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()


def run_algorithm(algorithm, maze_file, results_box):
    try:
        solver = MazeSolver(maze_file)
        start_time = time.time()

        if algorithm == "BFS":
            result = solver.bfs_search()
        elif algorithm == "DFS":
            result = solver.dfs_search()
        elif algorithm == "A*":
            result = solver.a_star_search()
        elif algorithm == "IDA*":
            result = solver.ida_star_search()
        elif algorithm == "IDS":
            result = solver.ids_search()
        elif algorithm == "RBFS":
            result = solver.rbfs_search()
        else:
            messagebox.showerror("Error", "Unknown algorithm selected.")
            return

        if result:
            elapsed_time = time.time() - start_time
            results_box.insert(
                tk.END,
                f"{algorithm}: Time={elapsed_time:.4f}s, Cost={result['path_cost']}, Nodes={result['expanded_nodes']}"
            )
            visualizer = MazeVisualizer(solver.maze, result['path'])
            visualizer.display_maze()
        else:
            messagebox.showinfo("No Solution", "No path could be found.")
            results_box.insert(tk.END, f"{algorithm}: No solution found.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def main():
    def browse_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)

    def start_algorithm(algo):
        maze_file = file_entry.get()
        if not maze_file:
            messagebox.showerror("Error", "Please select a maze file.")
            return
        run_algorithm(algo, maze_file, results_box)

    root = tk.Tk()
    root.title("Maze Solver")
    root.geometry("600x500")
    root.configure(bg="#f0f0f0")

    header = tk.Label(root, text="Maze Solver", font=("Arial", 20, "bold"), bg="#f0f0f0")
    header.pack(pady=10)

    file_frame = tk.Frame(root, bg="#f0f0f0")
    file_frame.pack(pady=10)

    file_label = tk.Label(file_frame, text="Maze CSV File:", font=("Arial", 12), bg="#f0f0f0")
    file_label.grid(row=0, column=0, padx=10, pady=5)
    file_entry = tk.Entry(file_frame, width=40, font=("Arial", 12))
    file_entry.grid(row=0, column=1, padx=10, pady=5)
    browse_button = tk.Button(file_frame, text="Browse", command=browse_file, bg="#007bff", fg="white", font=("Arial", 12))
    browse_button.grid(row=0, column=2, padx=10, pady=5)

    button_frame = tk.Frame(root, bg="#f0f0f0")
    button_frame.pack(pady=10)

    algorithms = ["BFS", "DFS", "A*", "IDA*", "IDS", "RBFS"]
    colors = ["#007bff", "#28a745", "#dc3545", "#ffc107", "#17a2b8", "#6f42c1"]
    for i, algo in enumerate(algorithms):
        btn = tk.Button(
            button_frame,
            text=algo,
            command=lambda a=algo: start_algorithm(a),
            bg=colors[i],
            fg="white",
            font=("Arial", 12, "bold"),
            width=10,
            height=2,
            relief="raised",
        )
        btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)

    results_label = tk.Label(root, text="Algorithm Results:", font=("Arial", 14, "bold"), bg="#f0f0f0")
    results_label.pack(pady=10)
    results_box = Listbox(root, width=70, height=10, font=("Arial", 12))
    results_box.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()