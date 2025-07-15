# 🧠 MNIST Digit Classification with CNN

This project demonstrates handwritten digit classification using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) with a Convolutional Neural Network (CNN) built in PyTorch.

The goal is to train a deep learning model to recognize digits (0–9) from 28×28 grayscale images. The notebook covers data preparation, model architecture, training loop, and evaluation.

---

---

## 🔍 Dataset

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset consists of:

* **60,000** training images
* **10,000** test images
* Each image is **28x28** pixels, grayscale, and represents a digit from 0 to 9.

---

## 🚀 Model Overview

The CNN architecture includes:

* Two convolutional layers with ReLU activations and max pooling
* A fully connected layer
* Dropout regularization to prevent overfitting
* Output layer with 10 neurons (one for each digit class)

---

## 📈 Training Results

* **Final Validation Accuracy**: \~98.5%
* **Loss**: Reduced significantly with good generalization
* **F1 Score**: High precision and recall balance

Example performance metrics (varies per run):

| Metric              | Value   |
| ------------------- | ------- |
| Final Train Loss    | \~0.17  |
| Final Val Loss      | \~0.03  |
| Validation Accuracy | \~98.5% |
| Validation F1 Score | \~0.985 |
| Test Accuracy       | \~99.3% |
| Test F1 Score       | \~0.993 |

---

## 🛠️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/mnist-cnn-classifier.git
cd mnist-cnn-classifier
```

2. Install dependencies (preferably in a virtual environment):

```bash
pip install torch torchvision matplotlib
```

3. Run the notebook:
   Open `Mnist.ipynb` in Jupyter or VS Code and run the cells step-by-step.

---

## 📊 Example Predictions

After training, the notebook includes visualizations of sample predictions showing the model's high accuracy on test images.

---

## 📌 Future Improvements

* Implement training with a learning rate scheduler
* Add confusion matrix and per-class accuracy
* Experiment with data augmentation
* Test on corrupted MNIST variants

---

## 📚 References

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [MNIST Database](http://yann.lecun.com/exdb/mnist/)
* [Deep Learning with PyTorch](https://pytorch.org/tutorials/)

