// Package main is the entry point for the gotorch library
package main

import (
	"fmt"
	"gotorch/model"
	"gotorch/tensor"
)

func main() {
	// Create input tensor with shape [2, 2] representing 2 samples with 2 features each
	inputs := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})

	// Create target tensor with shape [2, 1] representing desired outputs
	targets := tensor.NewTensor([][]float64{{5.0}, {11.0}})

	// Initialize a linear model with weights [1.0, 2.0] and bias [0.5]
	model := &model.Linear{
		Weights: []float64{1.0, 2.0},
		Biases:  []float64{0.5},
	}

	// Training hyperparameters
	epochs := 100       // Number of complete passes through the training data
	learningRate := 0.1 // Step size for gradient descent updates

	// Train the model using gradient descent
	model.Train(model, inputs, targets, epochs, learningRate)

	// Test the trained model on new data
	newInput := tensor.NewTensor([][]float64{{5.0, 6.0}})
	prediction := model.Sample(model, newInput)

	fmt.Println("Prediction:", prediction)
}
