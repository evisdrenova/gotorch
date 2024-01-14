package nn

import (
	"gotorch/tensor"
	"math"
)

const leaky_relu_constant = 0.01

// implements a ReLu activation function on each element in a tensor where f(x) = max(0,x)
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

// implements leaky_relu which is like relu but instead of zero'ing out anything less than 0, we multiply it by a small constant
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

func Sigmoid(t *tensor.Tensor) *tensor.Tensor {

	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = 1 / (1 + math.Exp(-t.Data[i]))
	}
	return &tensor.Tensor{Data: result, Shape: t.Shape}

}

func Tanh(t *tensor.Tensor) *tensor.Tensor {

	result := make([]float64, len(t.Data))
	for i := range t.Data {
		result[i] = (math.Exp(t.Data[i]-math.Exp(-t.Data[i])) / (math.Exp(t.Data[i]) + math.Exp(-t.Data[i])))
	}
	return &tensor.Tensor{Data: result, Shape: t.Shape}

}
