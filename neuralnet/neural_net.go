package nn

import "gotorch/tensor"

// implements a ReLu activation function on each element in a tensor
func ReLu(data *tensor.Tensor) []float64 {
	result := make([]float64, len(data.Data))
	for i, val := range data.Data {
		if val < 0 {
			result[i] = 0
		} else {
			result[i] = val
		}
	}
	return result
}
