// Package lf contains all of the loss functions
package lf

import (
	"gotorch/tensor"
	"math"
)

// implements mean squared error loss function; f(x,y) = average[((x1-y1)^2 + (x2-y2)^2  + ... (xn-yn)^2 )]
func MSELoss(input, target *tensor.Tensor) *tensor.Tensor {

	if len(input.Data) != len(target.Data) {
		panic("input and output tensors must have the same size")
	}

	var sum float64
	for i := range input.Data {
		diff := input.Data[i] - target.Data[i]
		sum += diff * diff
	}

	mse := sum / float64(len(input.Data))

	return &tensor.Tensor{Data: []float64{mse}, Shape: []int{1}}

}

// implements binary cross-entropy loss for binary class models
// quantifies the difference between the predicted probability distribution of the model and the actual distribution of the labels and returns a number between [0,1],with 0 being a perfect model
func BinaryCrossEntropyLoss(predications, target *tensor.Tensor) *tensor.Tensor {

	if len(predications.Data) != len(target.Data) {
		panic("input and output tensors must have the same size")
	}

	var sum float64

	for i := range predications.Data {
		p := predications.Data[i] //predicated probability
		y := target.Data[i]       // actual probability
		sum -= y*math.Log(p) + (1-y)*math.Log(1-p)
	}

	loss := sum / float64(len(predications.Data))

	return &tensor.Tensor{Data: []float64{loss}, Shape: []int{1}}

}

// implements categorical cross-entropy loss for multi-class models
// quantifies the difference between the predicted probability distribution of the model and the actual distribution of the labels and returns a number between [0,1],with 0 being a perfect mode
func CategoricalCrossEntropyLoss(predictions, target *tensor.Tensor) *tensor.Tensor {
	if len(predictions.Data) != len(target.Data) {
		panic("input and output tensors must have the same size")
	}

	var numClasses int
	var sum float64

	// Check if we are dealing with a vector or a multi-dimensional tensor
	if len(predictions.Shape) == 1 { // Vector case
		numClasses = len(predictions.Data)
		for c := 0; c < numClasses; c++ {
			sum -= target.Data[c] * math.Log(predictions.Data[c])
		}
	} else { // Multi-dimensional tensor case
		numClasses = predictions.Shape[1]
		for i := 0; i < len(predictions.Data); i += numClasses {
			for c := 0; c < numClasses; c++ {
				sum -= target.Data[i+c] * math.Log(predictions.Data[i+c])
			}
		}
	}

	loss := sum / float64(len(predictions.Data)/numClasses)

	return &tensor.Tensor{Data: []float64{loss}, Shape: []int{1}}

}
