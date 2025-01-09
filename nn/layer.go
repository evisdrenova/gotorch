package nn

import "gotorch/tensor"

type Layer interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(gradOutput *tensor.Tensor) *tensor.Tensor
	UpdateParams(learningRate float64)
}
