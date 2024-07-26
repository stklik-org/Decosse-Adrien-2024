import matplotlib.pyplot as plt

data = []

with open('compute/logs/J1_movements_waited.log', 'r') as file:
    for line in file:
        numbers = [float(number) for number in line.split()]
        data.append(numbers)

data = list(zip(*data))
plt.plot(data[0], label='x')
plt.plot(data[1], label='y')
plt.plot(data[2], label='z')
delta = []
for i in range(len(data[0])):
    delta.append((data[0][i]**2+data[1][i]**2+data[2][i]**2)**0.5)
plt.plot(delta, label='Δ')
plt.legend()
plt.show()

x_mean = 0
y_mean = 0
z_mean = 0

for i in range(len(data[0])):
    x_mean += data[0][i]
    y_mean += data[1][i]
    z_mean += data[2][i]
x_mean /= len(data[0])
y_mean /= len(data[1])
z_mean /= len(data[2])

x_sigma = 0
y_sigma = 0
z_sigma = 0
for i in range(len(data[0])):
    x_sigma += (data[0][i] - x_mean)**2
    y_sigma += (data[1][i] - y_mean)**2
    z_sigma += (data[2][i] - z_mean)**2
x_sigma = x_sigma**0.5
y_sigma = y_sigma**0.5
z_sigma = z_sigma**0.5

print(f"xmean={x_mean:.3f} with sigma={x_sigma:.3f}")
print(f"ymean={y_mean:.3f} with sigma={y_sigma:.3f}")
print(f"zmean={z_mean:.3f} with sigma={z_sigma:.3f}")