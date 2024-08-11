package nn

import (
	"gotorch/tensor"
	"math"
)

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
