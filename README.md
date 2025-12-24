# Neural Network From Scratch (NumPy Implementation)

A modular, deep learning framework built entirely from scratch using Python and NumPy. This project demonstrates a deep understanding of vector calculus, backpropagation, and matrix-based optimization.

Features
* Modular Architecture: Easily add or remove hidden layers.
* Custom Activations: Includes manual implementations of Sigmoid and Softmax.
* Manual Backpropagation: No Autograd or PyTorch. All gradients, including the Softmax Jacobian are calculated using manual matrix calculus.

Key Mathematical Implementations

Softmax Jacobian & Sigmoid  Binary Cross-Entropy
Instead of using shortcuts, this engine calculates the full relationship between logits and probabilities.
Gradient calcualtion: Activation and Layer Derivatives are calculated in forward prop. Calculates gradients per layer in back prop manually.

I wanted to understand what machine learning is. I wanted to know what weights and weight updation is. After studying the math behind it for a few weeks, and after practicing Linked lists and graphs a bit
I started building this framework that can handle Linear, Sigmoid, Relu and Softmax activations.
The project is very basic, and not optimized, but it predicts well in 1000 epochs. 
This project handles batch processing.
I have not converted the standardized predicted value back to actual scale yet.
I am yet to integrate my Adam, RMS, AdaGrad or Momentum optimizer functions to this project.
