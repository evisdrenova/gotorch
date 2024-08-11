package main

import (
	"fmt"
	"gotorch/model"
	"gotorch/tensor"
)

func main() {
	inputs := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	targets := tensor.NewTensor([][]float64{{5.0}, {11.0}})

	model := &model.Linear{
		Weights: []float64{1.0, 2.0},
		Biases:  []float64{0.5},
	}

	epochs := 100
	learningRate := 0.1

	// Train the model
	model.Train(model, inputs, targets, epochs, learningRate)

	// Sample the trained model
	newInput := tensor.NewTensor([][]float64{{5.0, 6.0}})
	prediction := model.Sample(model, newInput)

	fmt.Println("Prediction:", prediction)
}
