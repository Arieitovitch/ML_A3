# Task 1 part 3: have issues with overflow

- Attempted to use a standard stabilization technique : exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
- implemented large value clipping: 
```python
def softmax(self, z):
    # Clip z to prevent very large values
    z = np.clip(z, -700, 700)  # Avoid overflow in np.exp
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

```
- also attempted to use log-softmax

- Realized that we don't need clipping as much after doing z-score normalization (ie gaussian)
  - still need it when batch size is small
- Trying with smaller and smaller batch sizes. Added clipping back only on the softmax function
- 


# Task 2:


## Leaky Relu - vanishing gradient
1. Improper Weight Initialization
Standard Deviation Too Low: Initializing weights with a small standard deviation (e.g., 
0.01
0.01) can lead to small initial activations.
