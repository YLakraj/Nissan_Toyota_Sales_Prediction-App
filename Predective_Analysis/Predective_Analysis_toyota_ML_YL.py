
import matplotlib.pyplot as plt
import numpy as np


year = [2019, 2020, 2021, 2022,2023,2024]
unit_sold = [237091, 220675, 199308, 200204,227460,158317]
j = np.array(["2019", "2020", "2021", "2022","2023","2024"])
k = np.array([237091, 220675, 199308, 200204,227460,158317])

plt.bar(j, k, color = "green")
plt.show()
plt.bar(year,unit_sold,color = "pink")
plt.show()

print("hello values")