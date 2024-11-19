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
