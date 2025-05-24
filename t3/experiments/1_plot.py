import matplotlib
matplotlib.use('Agg')  # Use this before importing pyplot
import matplotlib.pyplot as plt
import os

# Path to data directory
data_dir = "experiments/horizontal_obj/"
filenames = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt") and not f.startswith("times"))
#filenames = [f for f in filenames if f.startswith("pos_") or f.startswith("neg_") or f.startswith("sample") or f.startswith("partitura")]

# Set up the plot
plt.figure(figsize=(12, 6))

# Loop through all files and plot each one
for file in filenames:
    filepath = os.path.join(data_dir, file)
    with open(filepath, "r") as f:
        data = f.read()

    angles = []
    values = []
    best = 0

    for line in data.strip().splitlines():
        try:
            parts = line.split(" - ")
            angle = float(parts[0].split(": ")[1])
            value = float(parts[1].split(": ")[1])
            angles.append(angle)
            values.append(value)
        except (IndexError, ValueError):
            print(f"Skipping malformed line: {line}")

    zipped = list(zip(angles, values))
    zipped.sort(key=lambda x: x[0])
    angles, values = zip(*zipped)

    best = max(values)
    best_index = values.index(best)
    best_angle = angles[best_index]

    # Mark the best point
    plt.plot(angles, values, linestyle='-', label=file[:-4] + ".png")
    plt.plot(best_angle, best, marker='o', color='black', markersize=6)
    #plt.plot(17, values[angles.index(17)], marker='o', color='black', markersize=6)
    plt.annotate(f"{best_angle:.1f}°", (best_angle, best), textcoords="offset points", xytext=(5,5), ha='left', fontsize=12, color='black')
    #plt.annotate(f"{17:.1f}°", (17, values[angles.index(17)]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=12, color='black')


# Finalize plot
plt.xlim(min(angles) - 1, max(angles) + 1)
plt.xticks([-90, -60, -30, 0, 30, 60, 90])
#plt.title("Projeção horizontal - Função Objetivo")
plt.xlabel("Ângulo (graus)", fontsize=14)
plt.ylabel("Valor da Função Objetivo", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("experiments/horizontal_obj/plot.png", dpi=450)