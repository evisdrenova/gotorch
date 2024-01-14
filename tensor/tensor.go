package tensor

import (
	"fmt"
	"gotorch/utils"
)

/*
The tensor package handles creating, updating and performing actions on tensors
*/

// Data holds the tensor data
// Shape defines how many dimensions the tensor has for ex. 2,3 for 2x3 matrix
type Tensor struct {
	Data  []float64
	Shape []int
}

// Creates a new tensor and returns a pointer to the tensor
func NewTensor(data interface{}, shape ...int) *Tensor {

	switch v := data.(type) {
	case int:
		return &Tensor{Data: []float64{float64(v)}, Shape: []int{1}}
	case int32:
		return &Tensor{Data: []float64{float64(v)}, Shape: []int{1}}
	case int64:
		return &Tensor{Data: []float64{float64(v)}, Shape: []int{1}}
	case float64:
		return &Tensor{Data: []float64{v}, Shape: []int{1}}
	case []float64:
		return &Tensor{Data: v, Shape: []int{len(v)}}
	case [][]float64:
		return newTensorFrom2DSlice(v)
	default:
		panic(fmt.Sprintf("unsupported type: %T", v))
	}
}

// newTensorFrom2DSlice handles creation of a tensor from a 2D slice
func newTensorFrom2DSlice(data [][]float64) *Tensor {
	if len(data) == 0 {
		return &Tensor{Data: []float64{}, Shape: []int{0, 0}}
	}

	numRows := len(data)
	numCols := len(data[0])
	flatData := make([]float64, 0, numRows*numCols)

	for _, row := range data {
		if len(row) != numCols {
			panic("rows of different lengths")
		}
		flatData = append(flatData, row...)
	}

	return &Tensor{Data: flatData, Shape: []int{numRows, numCols}}
}

// returns the number of dimensions for a tensor
func (t *Tensor) Dims() int {
	return len(t.Shape)
}

// adds two tensors together
// note: this does not currently support broadcasting to a common shape i.e. can't add a vector and a matrix
func Add(input, other *Tensor) (*Tensor, error) {

	if !utils.AreSlicesEqual(input.Shape, other.Shape) {
		return nil, fmt.Errorf("the two tensors must have the same shape in order to add them")
	}

	result := NewTensor(make([]float64, len(input.Data)), input.Shape...)

	for i := range input.Data {
		result.Data[i] = input.Data[i] + other.Data[i]
	}

	return result, nil
}
