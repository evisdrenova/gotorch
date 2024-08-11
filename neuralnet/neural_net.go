package nn

import (
	"gotorch/tensor"
	"math"
)

const leaky_relu_constant = 0.01

// implements a ReLu activation function on each element in a tensor where f(x) = max(0,x) which essentially zeros out any negative values
func ReLu(t *tensor.Tensor) *tensor.Tensor {
	result := make([]float64, len(t.Data))
	for i, val := range t.Data {
		if val < 0 {
			result[i] = 0
		} else {
			result[i] = val
		}
	}
	return &tensor.Tensor{Data: result, Shape: t.Shape}
}

// implements leaky_relu which is like relu but instead of zero'ing out anything less than 0, we multiply it by a small constant; f(x) = max((x*alpha), x)
func Leaky_ReLu(t *tensor.Tensor) *tensor.Tensor {
	result := make([]float64, len(t.Data))
	for i, val := range t.Data {
		if val < 0 {
			result[i] = val * leaky_relu_constant
		} else {
			result[i] = val
		}
	}
	return &tensor.Tensor{Data: result, Shape: t.Shape}
}

// sigmoid activation function: f(x) = 1/(1 + e^(-x))
func Sigmoid(t *tensor.Tensor) *tensor.Tensor {

	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = 1 / (1 + math.Exp(-t.Data[i]))
	}
	return &tensor.Tensor{Data: result, Shape: t.Shape}

}

// tanh activation function; f(x) = (e^(x) - e^(-x))/(e^(x) + e^(-x))
func Tanh(t *tensor.Tensor) *tensor.Tensor {

	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = (math.Exp(t.Data[i]-math.Exp(-t.Data[i])) / (math.Exp(t.Data[i]) + math.Exp(-t.Data[i])))
	}
	return &tensor.Tensor{Data: result, Shape: t.Shape}

}

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

// implements the softmax activation function which converts a tensor of float64s to a tensor of probability distributions
func SoftMax(t *tensor.Tensor) *tensor.Tensor {

	result := make([]float64, len(t.Data))

	var sum float64
	for _, value := range t.Data {
		sum += math.Exp(value)
	}

	for i, value := range t.Data {
		result[i] = math.Exp(value) / sum
	}

	return &tensor.Tensor{Data: result, Shape: t.Shape}
}
