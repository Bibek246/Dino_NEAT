import cv2
import numpy as np
import time
import neat
import os
import pickle
import matplotlib.pyplot as plt
from PIL import ImageGrab
import pyautogui

# Adjust game region (left, top, right, bottom) - Ensure correctness
game_region = (300, 400, 900, 550)

def capture_game():
    """Capture a grayscale screenshot of the game using PIL.ImageGrab."""
    try:
        left, top, right, bottom = game_region
        if right <= left or bottom <= top:
            raise ValueError("Invalid game region coordinates: right must be > left and bottom must be > top.")
        
        screenshot = ImageGrab.grab(bbox=game_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame
    except Exception as e:
        print(f"Error capturing game screenshot: {e}")
        return np.zeros((bottom - top, right - left), dtype=np.uint8)  # Return a blank frame

def detect_obstacle(frame):
    """Detect the nearest obstacle and return the distance."""
    _, thresh = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        obstacle = min(contours, key=lambda c: cv2.boundingRect(c)[0])
        x, _, w, _ = cv2.boundingRect(obstacle)
        return x - 50  # Adjust based on detection accuracy
    return 600  # Default large distance if no obstacle is found

def jump():
    """Make the Dino jump."""
    pyautogui.press("space")

def eval_genomes(genomes, config):
    """Evaluate genomes by playing the game."""
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        start_time = time.time()
        alive = True
        
        while alive:
            frame = capture_game()
            obstacle_dist = detect_obstacle(frame)
            inputs = [obstacle_dist / 600, 1]
            output = net.activate(inputs)
            
            if output[0] > 0.5:
                jump()
            
            time.sleep(0.1)
            new_frame = capture_game()
            if detect_obstacle(new_frame) < 50:  # Dino crashed
                alive = False
                
            fitness += time.time() - start_time
            genome.fitness = fitness

def plot_fitness(stats):
    """Plot and save the fitness evolution graph."""
    gen = range(len(stats.get_fitness_mean()))
    mean_fitness = stats.get_fitness_mean()
    best_fitness = stats.get_fitness_stat(max)
    
    plt.figure(figsize=(8, 6))
    plt.plot(gen, mean_fitness, label="Mean Fitness")
    plt.plot(gen, best_fitness, label="Best Fitness", linestyle="--", color="orange")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("NEAT AI Training Progress")
    plt.legend()
    plt.savefig("fitness_progress.png")  # Save the figure
    plt.show()

def run_neat(config_path):
    """Run the NEAT algorithm."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    winner = population.run(eval_genomes, 100)  # Increase generations for better AI
    print(f"Best AI found: {winner}")
    
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    
    plot_fitness(stats)

def play_with_trained_ai(config_path):
    """Use the trained AI to play the game."""
    with open("best_genome.pkl", "rb") as f:
        best_genome = pickle.load(f)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    
    print("Press 'Ctrl+C' to exit AI control.")
    try:
        while True:
            frame = capture_game()
            obstacle_dist = detect_obstacle(frame)
            inputs = [obstacle_dist / 600, 1]
            output = best_net.activate(inputs)
            
            if output[0] > 0.5:
                jump()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nAI stopped.")

if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), "config-feedforward.txt")
    run_neat(config_path)
    play_with_trained_ai(config_path)
