import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure images directory exists
SAVE_DIR = './images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def triangular_membership(x, a, l, r):
    if x < a - l:
        return 0
    elif a - l <= x <= a:
        return (x - (a - l)) / l
    elif a < x <= a + r:
        return (a + r - x) / r
    else:
        return 0

def plot_and_save(a1, l1, r1, a2, l2, r2, filename):
    print(f"Generating {filename} with parameters: A=({a1},{l1},{r1}), B=({a2},{l2},{r2})")
    x_values = np.linspace(0, 8, 400)
    y_values_1 = [triangular_membership(x, a1, l1, r1) for x in x_values]
    y_values_2 = [triangular_membership(x, a2, l2, r2) for x in x_values]

    plt.figure(figsize=(10, 8))
    plt.plot(x_values, y_values_1, label='A(x)', color='lightskyblue', linewidth=5)
    plt.plot(x_values, y_values_2, label='B(x)', color='lightsalmon', linewidth=5)

    plt.xlabel('', fontsize=30)
    plt.ylabel('', rotation=0, verticalalignment='center', fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=20)

    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")

# List of parameters from the notebook cells
# Cell 2: a1, l1, r1 = 3, 2, 2; a2, l2, r2 = 5, 1, 1
# Cell 3: a1, l1, r1 = 3, 2, 2; a2, l2, r2 = 7, 1, 1
# Cell 4: a1, l1, r1 = 4, 2, 2; a2, l2, r2 = 4, 1, 1
# Cell 5: a1, l1, r1 = 4, 2, 2; a2, l2, r2 = 4, 2, 2
# Cell 6: a1, l1, r1 = 3, 1, 1; a2, l2, r2 = 4, 3, 3
# Cell 7: a1, l1, r1 = 4, 3, 3; a2, l2, r2 = 5, 1, 1

params_list = [
    (3, 2, 2, 5, 1, 1, 'distribution_1.png'),
    (3, 2, 2, 7, 1, 1, 'distribution_2.png'),
    (4, 2, 2, 4, 1, 1, 'distribution_3.png'),
    (4, 2, 2, 4, 2, 2, 'distribution_4.png'),
    (3, 1, 1, 4, 3, 3, 'distribution_5.png'),
    (4, 3, 3, 5, 1, 1, 'distribution_6.png')
]

if __name__ == "__main__":
    for params in params_list:
        plot_and_save(*params)
    print("All plots generated and saved successfully.")
