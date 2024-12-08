import numpy as np
import matplotlib.pyplot as plt

raw_data = np.load('IMG_9939.npy')
print('Loaded array of size', raw_data.shape)
print('The pens, from top to bottom, are red, green and blue')

# Visualize the raw data
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.title('Raw Data Full View')
plt.imshow(raw_data)
plt.colorbar()

plt.subplot(1,2,2)
plt.title('Raw Data Histogram')
plt.hist(raw_data.flatten(), bins=200, color='gray', alpha=0.7)
plt.title('Pixel Value Distribution')

plt.tight_layout()
plt.show()

# - inspect blocks of 2x2 and find pattern in different regions 
# - top left, top right, bottom left, bottom right 
# - green would have the highest values, blue the lowest
def pattern(arr, x, y):
    print(f"pattern NxN at [{x},{y}]:")
    print(arr[x:x+4, y:y+4])

# Check a few regions
print("\nSample NxN patterns:")
pattern(raw_data, 100, 100)
pattern(raw_data, 300, 300)
pattern(raw_data, 500, 500)
pattern(raw_data, 700, 700)