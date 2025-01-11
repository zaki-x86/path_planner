from path_planner import InteractiveMazeGraph, load_maze_from_file

def main():
    # Load maze from file
    maze = load_maze_from_file('maze.txt')
    
    if maze is not None:
        interactive_maze = InteractiveMazeGraph(maze)
        interactive_maze.run()
        
    else:
        print("Failed to load maze. Please check the maze.txt file.")

if __name__ == "__main__":
    main()
