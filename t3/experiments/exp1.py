import matplotlib.pyplot as plt
import os

# Raw data as a multi-line string
filenames = os.listdir("experiments/horizontal_obj/")
for file in filenames:
    if file.endswith(".txt"):
        with open(os.path.join("experiments/horizontal_obj/", file), "r") as f:
            data = f.read()

    # Parse data into lists
    angles = []
    values = []

    for line in data.strip().splitlines():
        try:
            parts = line.split(" - ")
            angle = float(parts[0].split(": ")[1])
            value = float(parts[1].split(": ")[1])
            angles.append(angle)
            values.append(value)
        except (IndexError, ValueError):
            print(f"Skipping malformed line: {line}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(angles, values, marker='o', linestyle='-', color='steelblue')
    plt.title("Objective Value vs Angle")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Objective Value")
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.close('all')
    plt.savefig(f"experiments/horizontal_obj/{file.split('.')[0]}.png")
