import logging
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Button, RadioButtons

logger = logging.getLogger(__name__)


def configure_logging(logger: logging.Logger):
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def load_maze_from_file(filename):
    """Load maze from text file with +, -, | format"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Remove any trailing newlines and get maze dimensions
        lines = [line.rstrip() for line in lines]
        height = len(lines)
        width = len(lines[0])

        # Create a binary maze array (True for paths, False for walls)
        maze = np.zeros((height, width), dtype=bool)

        # Convert the ASCII maze to binary
        # Spaces are paths (True), everything else is wall (False)
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                maze[i, j] = (char == ' ')

        # Convert to standard 0/1 format (0 for path, 1 for wall)
        return ~maze

    except FileNotFoundError:
        logger.error(f"Error: File '{filename}' not found")
        return None
    except Exception as e:
        logger.error(f"Error reading maze file: {e}")
        return None


class InteractiveMazeGraph:
    def __init__(self, maze):
        configure_logging(logger)
        logger.info('Initializing InteractiveMazeGraph')
        self.maze: np.ndarray = maze
        if self.maze is None:
            logger.error('Maze is empty, exiting')
            exit(1)
        self.height: int = len(maze)
        self.width: int = len(maze[0])
        logger.info('Maze dimensions: %s x %s', self.height, self.width)
        # Directions:
        # Right: (0, 1), Down: (1, 0), Left: (0, -1), Up: (-1, 0)
        self.directions: list[tuple[int, int]] = [
            (0, 1), (1, 0), (0, -1), (-1, 0)]
        self.graph: nx.Graph = self._create_graph()

        # Initialize variables
        self.fig: Figure = None
        self.ax: Axes = None
        self.start_point: tuple[int, int] = None
        self.end_point: tuple[int, int] = None
        self.current_path: list[tuple[int, int]] = None
        self.clicks: int = 0
        self.algorithm: str = 'bellman-ford'

        # Create the figure and plot
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_title('Interactive Maze Solver')
        self.ax.set_label('Click to set start and end points')
        plt.subplots_adjust(bottom=0.1, left=0.1)

        # Add reset button
        self.reset_button_ax: Axes = plt.axes(
            [0.9, 0.05, 0.1, 0.075], label='Reset')
        self.reset_button: Button = Button(
            ax=self.reset_button_ax,
            label='Reset',
            color='lightgoldenrodyellow')
        self.reset_button.on_clicked(self.reset)

        # Add algorithm selection radio buttons
        self.radio_ax: Axes = plt.axes(
            [0.05, 0.05, 0.2, 0.2], label='Algorithm Selection')
        self.radio: RadioButtons = RadioButtons(
            ax=self.radio_ax,
            labels=('bellman-ford', 'dijkstra', 'a_star', 'dfs', 'bfs'),
            active=0,
            activecolor='green')
        self.radio.on_clicked(self.set_algorithm)

        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Initial plot
        self.update_plot()

    def _create_graph(self):
        """Convert maze to NetworkX graph"""
        logger.debug('Creating nx graph from maze')
        G = nx.Graph()

        # Add nodes for all walkable cells
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:  # if it's a path
                    G.add_node((i, j))

        # Add edges between adjacent walkable cells
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    for drow, dcol in self.directions:
                        ni, nj = i + drow, j + dcol
                        if (0 <= ni < self.height and
                            0 <= nj < self.width and
                                self.maze[ni, nj] == 0):
                            G.add_edge((i, j), (ni, nj), weight=1)

        logger.info(f'Graph created with {len(G.nodes())} vertices.')

        return G

    def find_path(self, start, end):
        """Find path using selected algorithm"""
        logger.info('Calculating path from %s to %s', start, end)
        try:
            if self.algorithm == 'bellman-ford':
                return nx.shortest_path(self.graph, start, end, method='bellman-ford')
            elif self.algorithm == 'dijkstra':
                return nx.dijkstra_path(self.graph, start, end)
            elif self.algorithm == 'a_star':
                return nx.astar_path(self.graph, start, end,
                                     heuristic=self.manhattan_distance)
            elif self.algorithm == 'bfs':
                return nx.bfs_tree(self.graph, start)
            elif self.algorithm == 'dfs':
                return nx.dfs_tree(self.graph, start)
            else:
                return None
        except nx.NodeNotFound as e:
            logger.error(f"Start node selection error: {e}")
            return None
        except ValueError as e:
            logger.error(f"Method selection error: {e}")
            return None
        except nx.NetworkXNoPath as e:
            logger.error(f"No path found: {e}")
            return None

    def manhattan_distance(self, pos1, pos2):
        """Manhattan distance heuristic for A*"""
        # logger.debug('Calculating Manhattan distance between %s and %s',
        #              pos1, pos2)
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_nearest_node(self, x, y):
        """Find the nearest valid node to the clicked position"""
        logger.debug('Getting nearest node to (%s, %s)', x, y)
        x, y = int(round(x)), int(round(y))
        if (0 <= y < self.height and
            0 <= x < self.width and
                self.maze[y, x] == 0):
            return (y, x)
        return None

    def set_algorithm(self, label):
        """Change the pathfinding algorithm"""
        logger.info('Algorithm changed to %s', label)
        self.algorithm = label
        if self.start_point and self.end_point:
            start_time = time.time()
            self.current_path = self.find_path(
                self.start_point, self.end_point)
            end_time = time.time()
            logger.info(
                f"Time taken by {self.algorithm}: {end_time - start_time:.4f} seconds")
            self.update_plot()

    def on_click(self, event):
        """Handle mouse clicks"""
        logger.debug('Click event: %s', event)
        if event.inaxes != self.ax:
            return

        node = self.get_nearest_node(event.xdata, event.ydata)
        if node is None:
            logger.debug('Clicked outside of maze')
            return

        logger.info('Clicked on node: %s', node)

        if self.clicks == 0:
            self.start_point = node
            self.clicks = 1
            logger.info('Start point selected: %s', self.start_point)
        elif self.clicks == 1:
            self.end_point = node
            logger.info('End point selected: %s', self.end_point)
            self.current_path = self.find_path(
                self.start_point, self.end_point)
            self.clicks = 2

        self.update_plot()

    def reset(self, event: MouseEvent):
        """Reset the selection"""
        logger.debug('Resetting selection')
        self.start_point = None
        self.end_point = None
        self.current_path = None
        self.clicks = 0
        self.update_plot()

    def update_plot(self):
        """Update the plot with current state"""
        logger.debug('Updating plot')
        
        logger.debug('Clearing plot')
        self.ax.clear()

        # Draw maze
        logger.debug('Drawing maze')
        self.ax.imshow(self.maze, cmap='binary')

        # Draw graph structure
        pos = {node: (node[1], node[0]) for node in self.graph.nodes()}

        logger.debug('Drawing graph nodes and edges')
        nx.draw_networkx_nodes(
            G=self.graph,
            pos=pos,
            node_color='lightblue',
            node_size=10,
            ax=self.ax)
        nx.draw_networkx_edges(
            G=self.graph,
            pos=pos,
            edge_color='lightblue',
            width=0.5,
            ax=self.ax)

        # Draw start point
        if self.start_point:
            logger.debug('Drawing start point: %s', self.start_point)
            self.ax.plot(self.start_point[1], self.start_point[0],
                         'go', markersize=10, label='Start')

        # Draw end point
        if self.end_point:
            logger.debug('Drawing end point: %s', self.end_point)
            self.ax.plot(self.end_point[1], self.end_point[0],
                         'ro', markersize=10, label='End')

        # Draw path
        if self.current_path:
            path = np.array(self.current_path)
            logger.debug('Drawing path with length: %s', len(self.current_path))
            self.ax.plot(path[:, 1], path[:, 0], 'r-',
                         linewidth=3, label=f'Path ({self.algorithm})')

        self.ax.grid(True)
        self.ax.legend()
        self.fig.canvas.draw()

    def run(self):
        """Render the plot"""
        # Set the window size (in pixels)
        plt.rcParams['figure.figsize'] = [40, 40]  # Width, height in inches
        plt.rcParams['figure.dpi'] = 100  # Dots per inch
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')  # Works on Windows
        except:
            try:
                mng.window.showMaximized()  # Works on Qt
            except:
                try:
                    mng.frame.Maximize(True)  # Works on WX
                except:
                    pass  # Fall back to default size
        plt.show()
