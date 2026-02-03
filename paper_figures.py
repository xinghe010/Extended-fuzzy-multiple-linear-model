import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set image save directory
SAVE_DIR = './images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def plot_example_3():
    """
    Plot example_3 symmetric triangular 1st-order data
    """
    print("Plotting example_3 data...")
    try:
        x = pd.read_csv('./data/example_3_symmetric_triangular.csv')['x']
        y = pd.read_csv('./data/example_3_symmetric_triangular.csv')['y']
        y1 = pd.read_csv('./data/example_3_M_z.csv')['y']
        y_p = pd.read_csv('./results/example_3_symmetric_triangular_estimates.csv')['y_p']

        plt.figure(figsize=(25, 15))

        # Scatter plot and Line plot
        plt.scatter(x, y, color='grey', label='The central value of the observations', s=200)
        plt.plot(x, y1, color='lightsalmon', linestyle='-', label='The central value of the estimated values from the $M_{Zhang}$', linewidth=3)
        plt.plot(x, y_p, color='lightskyblue', linestyle='-', label='The central value of the estimated values from the $M$', linewidth=3)

        plt.xlabel('x', fontsize=30)
        plt.ylabel('y', rotation=0, verticalalignment='center', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        
        save_path = os.path.join(SAVE_DIR, 'example_3_plot.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
    except Exception as e:
        print(f"Error plotting example_3: {e}")

def plot_shanghai_zhou2018_y():
    """
    Plot shanghai_zhou2018 y value comparison
    """
    print("Plotting shanghai_zhou2018 y value comparison...")
    try:
        y = pd.read_csv('./data/shanghai_zhou2018.csv')['y']
        y_MM = pd.read_csv('./results/shanghai_zhou2018_MM_estimates.csv')['y_p']
        y_M = pd.read_csv('./results/shanghai_zhou2018_estimates.csv')['y_p']

        plt.figure(figsize=(25, 15))

        plt.plot(y, y, marker='s', color='grey', label='The baseline representing the central value of the observations', linewidth=4)
        plt.scatter(y, y_MM, marker='o', color='lightsalmon', label='The central value of the estimated values from the $MM$', s=300)
        plt.scatter(y, y_M, marker='^', color='lightskyblue', label='The central value of the estimated values from the $M$', s=300)

        plt.xlabel('y', fontsize=30)
        plt.ylabel('y', rotation=0, verticalalignment='center', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        
        save_path = os.path.join(SAVE_DIR, 'shanghai_zhou2018_y.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
    except Exception as e:
        print(f"Error plotting shanghai_zhou2018 y: {e}")

def plot_shanghai_zhou2018_ly():
    """
    Plot shanghai_zhou2018 ly (left spread) comparison
    """
    print("Plotting shanghai_zhou2018 ly comparison...")
    try:
        ly = pd.read_csv('./data/shanghai_zhou2018.csv')['l']
        ly_MM = pd.read_csv('./results/shanghai_zhou2018_MM_estimates.csv')['ly_p']
        ly_M = pd.read_csv('./results/shanghai_zhou2018_estimates.csv')['ly_p']

        plt.figure(figsize=(25, 18))

        plt.plot(ly, ly, marker='s', color='grey', label='The baseline representing the left spread value of the observations', linewidth=4)
        plt.scatter(ly, ly_MM, marker='o', color='lightsalmon', label='The left spread value of the estimated values from the $MM$', s=350)
        plt.scatter(ly, ly_M, marker='^', color='lightskyblue', label='The left spread value of the estimated values from the $M$', s=350)

        plt.xlabel('$l_{y}$', fontsize=30)
        plt.ylabel('$l_{y}$', rotation=0, verticalalignment='center', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        
        save_path = os.path.join(SAVE_DIR, 'shanghai_zhou2018_ly.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
    except Exception as e:
        print(f"Error plotting shanghai_zhou2018 ly: {e}")

def plot_shanghai_zhou2018_ry():
    """
    Plot shanghai_zhou2018 ry (right spread) comparison
    """
    print("Plotting shanghai_zhou2018 ry comparison...")
    try:
        ry = pd.read_csv('./data/shanghai_zhou2018.csv')['r']
        ry_MM = pd.read_csv('./results/shanghai_zhou2018_MM_estimates.csv')['ry_p']
        ry_M = pd.read_csv('./results/shanghai_zhou2018_estimates.csv')['ry_p']

        plt.figure(figsize=(26, 18))

        plt.plot(ry, ry, marker='s', color='grey', label='The baseline representing the right spread value of the observations', linewidth=4)
        plt.scatter(ry, ry_MM, marker='o', color='lightsalmon', label='The right spread value of the estimated values from the $MM$', s=350)
        plt.scatter(ry, ry_M, marker='^', color='lightskyblue', label='The right spread value of the estimated values from the $M$', s=350)

        plt.xlabel('$r_{y}$', fontsize=30)
        plt.ylabel('$r_{y}$', rotation=0, verticalalignment='center', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        
        save_path = os.path.join(SAVE_DIR, 'shanghai_zhou2018_ry.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
    except Exception as e:
        print(f"Error plotting shanghai_zhou2018 ry: {e}")

if __name__ == "__main__":
    print("Starting to generate paper figures...")
    plot_example_3()
    plot_shanghai_zhou2018_y()
    plot_shanghai_zhou2018_ly()
    plot_shanghai_zhou2018_ry()
    print("All figures generated.")
