# Dino NEAT AI

This project trains a NEAT-based AI to play the Google Chrome Dino game by detecting obstacles and jumping at the right time.

## Features
- Uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve a neural network.
- Captures the game screen using `PIL.ImageGrab`.
- Detects obstacles using OpenCV.
- Controls the Dino using `pyautogui`.
- Saves the best-performing genome for later use.
- Plots training progress using Matplotlib.

## Prerequisites
Ensure you have the following installed:

- Python 3.7+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- NEAT-Python (`pip install neat-python`)
- PyAutoGUI (`pip install pyautogui`)
- Matplotlib (`pip install matplotlib`)
- Pillow (`pip install pillow`)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Dino-NEAT-AI.git
   cd Dino-NEAT-AI
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Configuration
This project uses a `config-feedforward.txt` file to configure the NEAT algorithm. It includes settings for genome mutation, population size, and fitness evaluation.

## How to Run
### Train the AI
Run the following command to start training:
```sh
python dino_neat.py
```
The AI will evolve over 100 generations, trying to improve its gameplay performance.

### Play Using the Trained AI
After training, use the best-performing AI to play automatically:
```sh
python dino_neat.py --play
```

## NEAT Configuration Details
The `config-feedforward.txt` file contains:
- **Population size**: Number of AI in each generation.
- **Mutation rates**: Probabilities for changing activation functions, adding/removing nodes, etc.
- **Fitness function**: The AI is rewarded for staying alive longer.
- **Neural network structure**: Number of input and output nodes.

## Training Visualization
A fitness plot is generated at the end of training, showing:
- Mean fitness across generations.
- Best-performing genome's fitness.

The plot is saved as `fitness_progress.png`.

## Saving & Loading Models
The best AI genome is saved in `best_genome.pkl`. This allows the AI to play without retraining.

## Troubleshooting
### Game Region Detection Issues
If the AI is not detecting obstacles correctly:
- Adjust `game_region` in `dino_neat.py` to match your screen resolution.

### Slow Performance
- Lower the population size in `config-feedforward.txt`.
- Reduce the number of generations.

### AI Not Jumping
- Ensure `pyautogui` has permission to simulate key presses.
- Check if `obstacle_dist` is being detected correctly.

## License
This project is licensed under the MIT License. Feel free to modify and distribute.

## Acknowledgments
- Inspired by the Google Dino game.
- NEAT-Python library for evolution-based AI.

Happy training!

