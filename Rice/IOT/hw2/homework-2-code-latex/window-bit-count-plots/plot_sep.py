import matplotlib.pyplot as plt
from collections import defaultdict

data = defaultdict(lambda: {'x': [], 'y_tput': [], 'y_mem': []})
with open('/Users/apple/Coded/Rice/IOT/hw2/homework-2-code-released/window-bit-count-plots/results.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            alg, w, tput, mem = parts
            data[alg]['x'].append(int(w))
            data[alg]['y_tput'].append(float(tput))
            data[alg]['y_mem'].append(float(mem))

# Throughput plot
plt.figure(figsize=(10, 5))
for alg, values in data.items():
    # Sort by window size for clean lines
    sorted_pairs = sorted(zip(values['x'], values['y_tput']))
    x, y = zip(*sorted_pairs)
    plt.plot(x, y, marker='o', label=alg)

plt.xscale('log')
plt.yscale('log')
plt.title('Throughput vs Window Size')
plt.xlabel('Window Size')
plt.ylabel('Throughput (items/sec)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('/Users/apple/Coded/Rice/IOT/hw2/throughput.pdf')
plt.close()

# Memory plot
plt.figure(figsize=(10, 5))
for alg, values in data.items():
    sorted_pairs = sorted(zip(values['x'], values['y_mem']))
    x, y = zip(*sorted_pairs)
    plt.plot(x, y, marker='o', label=alg)

plt.xscale('log')
plt.yscale('log')
plt.title('Memory Footprint vs Window Size')
plt.xlabel('Window Size')
plt.ylabel('Memory (bytes)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('/Users/apple/Coded/Rice/IOT/hw2/memory_footprint.pdf')
plt.close()
