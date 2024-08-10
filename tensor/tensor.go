package tensor

import (
	"fmt"
	"gotorch/utils"
	"math/rand"
	"strings"
)

/*
The tensor package handles creating, updating and performing actions on tensors.
We normalize all tensors into a flattened 1D tensor to make it easy to add/multiple/etc. the tensors. We retain the original shape by storing the Shape in the tensor.Shape struct. We can always re-construct the tensors original shape then.
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
		if len(shape) == 0 {
			shape = []int{len(v)}
		}
		return &Tensor{Data: v, Shape: shape}
	case [][]float64:
		return newTensorFrom2DSlice(v)
	default:
		panic(fmt.Sprintf("unsupported type: %T", v))
	}
}

// newTensorFrom2DSlice handles creation of a tensor from a 2D slice by flattening it into a 1D tensor which makes it easier to work with
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

// Adds two tensors together
// note: this does not currently support broadcasting to a common shape i.e. can't add a vector and a matrix
func Add(t1, t2 *Tensor) (*Tensor, error) {

	if !utils.AreSlicesEqual(t1.Shape, t2.Shape) {
		return nil, fmt.Errorf("the two tensors must have the same shape in order to add them")
	}

	result := NewTensor(make([]float64, len(t1.Data)), t1.Shape...)

	// since the two tensors are the same shape we can just iterate over one and then index into both tensors
	for i := range t1.Data {
		result.Data[i] = t1.Data[i] + t2.Data[i]
	}

	return result, nil
}

// Subtracts two tensors
func Subtract(t1, t2 *Tensor) (*Tensor, error) {

	if !utils.AreSlicesEqual(t1.Shape, t2.Shape) {
		return nil, fmt.Errorf("the two tensors must have the same shape in order to subtract them")
	}

	result := NewTensor(make([]float64, len(t1.Data)), t2.Shape...)

	for i := range t1.Data {
		result.Data[i] = t1.Data[i] - t2.Data[i]
	}

	return result, nil

}

// Multiplies two tensors
func Multiply(t1, t2 *Tensor) (*Tensor, error) {

	if !utils.AreSlicesEqual(t1.Shape, t2.Shape) {
		return nil, fmt.Errorf("the two tensors must have the same shape in order to multiply them")
	}

	result := NewTensor(make([]float64, len(t1.Data)), t1.Shape...)

	for i := range t1.Data {
		result.Data[i] = t1.Data[i] * t2.Data[i]
	}

	return result, nil

}

// Divides two tensors
func Divide(t1, t2 *Tensor) (*Tensor, error) {

	if !utils.AreSlicesEqual(t1.Shape, t2.Shape) {
		return nil, fmt.Errorf("the two tensors must have the same shape in order to divide them")
	}

	result := NewTensor(make([]float64, len(t1.Data)), t1.Shape...)

	for i := range t1.Data {
		result.Data[i] = t1.Data[i] / t2.Data[i]
	}

	return result, nil

}

// Returns a tensor filled with random numbers from a uniform distribution on the interval
// rows determines how many rows are in the tensor
// columns determines how many columns are in the tensor
func Rand(rows, columns int) (*Tensor, error) {

	if rows < 1 || columns < 1 {
		return nil, fmt.Errorf("rows or columns must be greater than 0")
	}

	flatData := make([]float64, 0, rows*columns)

	for i := 0; i < rows; i++ {
		for i := 0; i < columns; i++ {
			flatData = append(flatData, rand.Float64())
		}
	}

	return &Tensor{Data: flatData, Shape: []int{rows, columns}}, nil

}

// Given a tensor, FormatTensor will return the tensor in the right shape according to the shape property. For example, if the shape of the tensor is []int{2 3},
// meaning that is a 2x3 matrix, this function will return a multi-dimensional array with two sub arrays, each with three elements, essentially the expanded format of the tensor
// honestly this is ugly but whatever for now, its just meant as a sanity check
func FormatTensor(t *Tensor) string {

	switch {
	case len(t.Shape) == 1:
		var result strings.Builder
		for i, val := range t.Data {
			if i > 0 {
				result.WriteString(",")
			}
			result.WriteString(fmt.Sprintf("%.4f", val))
		}
		return "[" + result.String() + "]"
	case len(t.Shape) == 2:
		var result strings.Builder
		rowLength := t.Shape[1]

		result.WriteString("[\n")
		for i, val := range t.Data {
			if i%rowLength == 0 {
				if i != 0 {
					result.WriteString("],\n")
				}
				result.WriteString("  [")
			} else {
				result.WriteString(",")
			}
			result.WriteString(fmt.Sprintf("%.4f", val))
		}
		result.WriteString("]\n]")

		return result.String()

	case len(t.Shape) > 2:
		return formatTensorRecursive(t.Data, t.Shape, 0)
	default:
		return "Unable to format tensor"
	}

}

func formatTensorRecursive(data []float64, shape []int, level int) string {
	if len(shape) == 0 {
		return fmt.Sprintf("%.4f", data[0])
	}

	var result strings.Builder
	if level != 0 {
		result.WriteString("\n" + strings.Repeat(" ", level*2)) // Indentation for readability
	}
	result.WriteString("[")

	numElements := shape[0]
	subShape := shape[1:]
	subSize := 1
	for _, s := range subShape {
		subSize *= s
	}

	for i := 0; i < numElements; i++ {
		startIndex := i * subSize
		endIndex := startIndex + subSize
		result.WriteString(formatTensorRecursive(data[startIndex:endIndex], subShape, level+1))

		if i < numElements-1 {
			result.WriteString(",")

		}
	}

	result.WriteString("]")

	return result.String()
}
