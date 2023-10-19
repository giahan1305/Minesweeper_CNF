# Minesweeper_CNF
# Minesweeper Solver

This repository contains a Python script for solving Minesweeper puzzles represented as matrices. The script offers multiple solving algorithms, including brute force, backtracking, PySAT solver, and a heuristic resolution approach.

## Code Structure

The code is organized into several classes and functions:

1. **`priorityQueue`**: Implements a priority queue for efficient handling of items based on priority.
2. **`Cell`**: Represents a cell in the Minesweeper grid with row, column, and state attributes.
3. Various helper functions for generating Minesweeper CNF, clause resolution, and matrix operations.
4. **Solver Functions**:
    - `booleanLogic`: Evaluates logical expressions based on cell states.
    - `checkCNF_Brute_force`: A brute-force Minesweeper solver.
    - `checkCNF_backtracking`: A backtracking Minesweeper solver.
    - `is_satisfied`: Checks if Minesweeper clauses are satisfied.
    - `change_matrix`: Updates the matrix based on the solved Minesweeper problem.
    - `update8Square`: Updates labels in a cell's neighborhood.
    - `matrixBombGenerator`: Generates a Minesweeper matrix with mines and labels.
    - `convertLiteralToInt`: Converts a Minesweeper clause to an integer.
    - `formStrKB`: Converts Minesweeper clauses to a priority queue for PySAT solving.
    - `resolve`: Resolves two Minesweeper clauses.
    - `resolutionRefutation`: Performs resolution refutation to solve Minesweeper problems.
    - `HeuristicResolution`: Uses a heuristic resolution approach to solve Minesweeper problems.

## Usage

1. The script reads an input Minesweeper matrix from an input file.
2. The user can choose a solving algorithm (brute force, backtracking, PySAT solver, or heuristic resolution).
3. The script attempts to find the solution and prints the result.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required Python libraries: `numpy`, `pysat`, and `Glucose3`.
3. Run the script, providing an input file containing the Minesweeper matrix.

Enjoy solving Minesweeper puzzles!
