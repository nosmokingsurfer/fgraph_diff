import glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("./out/*_error.txt")

fig = plt.figure(figsize=(10,10))


for f in files:
    print(f)
    err = np.genfromtxt(f)

    plt.plot(err)


plt.show()
