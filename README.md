# Neural Network from Scratch: MNIST Digit Recognition

This project implements a simple neural network **from scratch using only NumPy** to recognize handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
It demonstrates the basics of forward and backward propagation, parameter updates, and prediction—without using any deep learning frameworks.

---

## What the Project Is

- **A beginner-friendly, fully-commented implementation of a neural network for digit classification.**
- **No TensorFlow, PyTorch, or Keras—just NumPy and basic Python.**
- **Trains on the MNIST dataset (CSV format) and predicts digits from images.**
- **Includes visualization of predictions.**

---

## How to Run It

1. **Clone or download this repository.**
2. **Download the MNIST training data in CSV format** (see below).
3. **Install requirements:**
    ```bash
    pip install numpy pandas matplotlib
    ```
4. **Run the script:**
    ```bash
    python digirecog.py
    ```

---

## Example Usage/Output

When you run the script, you will see output like:

```
Iteration: 0
Accuracy: 0.0982
Loss: 2.3025
...
Iteration: 990
Accuracy: 0.9123
Loss: 0.2854
Dev accuracy: 0.915
Prediction:  [7]
Label:  7
<shows digit image>
Prediction:  [2]
Label:  2
<shows digit image>
...
```

The script will also display a few sample digit images with their predicted and true labels.

---

## Dataset Requirements

- **You need the MNIST training data in CSV format.**
- You can download it from [Kaggle: Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data).
- Place `train.csv` in a `data` folder in the same directory as `digirecog.py`:
    ```
    your_project/
      digirecog.py
      data/
        train.csv
    ```

---

## Notes

- This project is for educational purposes and demonstrates the fundamentals of neural networks.
- For best results, use the full MNIST training set (42,000+ samples).
- You can adjust the number of training iterations and learning rate in the script.

---

**Enjoy learning how neural networks work under