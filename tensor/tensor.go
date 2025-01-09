// Package tensor implements multi-dimensional arrays and tensor operations
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

// Tensor represents a multi-dimensional array with shape information.
// Data is stored in a flattened 1D slice for efficient operations,
// while Shape maintains the tensor's dimensional information.
type Tensor struct {
	Data  []float64 // Flattened array of tensor elements
	Shape []int     // Dimensions of the tensor (e.g., [2,3] for 2x3 matrix)
}

// NewTensor creates a tensor from various input types.
// Supports scalars, vectors, matrices, and n-dimensional arrays.
// Shape can be explicitly specified or inferred from input structure.
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

// newTensorFrom2DSlice creates a tensor from a 2D slice by flattening it.
// Verifies that all rows have the same length to ensure valid matrix structure.
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

// Dims returns the number of dimensions of the tensor.
// For example: scalar = 1, vector = 1, matrix = 2, etc.
func (t *Tensor) Dims() int {
	return len(t.Shape)
}

// Add performs element-wise addition of two tensors.
// Tensors must have identical shapes for the operation to succeed.
// Returns a new tensor containing the sum and any error that occurred.
func Add(t1, t2 *Tensor) (*Tensor, error) {
	if !utils.AreSlicesEqual(t1.Shape, t2.Shape) {
		return nil, fmt.Errorf("the two tensors must have the same shape in order to add them")
	}

	result := NewTensor(make([]float64, len(t1.Data)), t1.Shape...)

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
	// a shape of 1 means that it's already flat, so we can just write it out
	case len(t.Shape) == 1:
		var result strings.Builder
		for i, val := range t.Data {
			if i > 0 {
				result.WriteString(",")
			}
			result.WriteString(fmt.Sprintf("%.4f", val))
		}
		return "[" + result.String() + "]"
		//a shape of 2 means that it's not a flat tensor and has shape so we need to iterate through it
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

// Dot performs the dot product of two tensors
// For vectors: it's the sum of element-wise products
// For matrices: it's matrix multiplication where (AB)ij = sum(Aik * Bkj)
func Dot(t1, t2 *Tensor) (*Tensor, error) {
	// Case 1: Vector dot Vector (1D * 1D)
	if len(t1.Shape) == 1 && len(t2.Shape) == 1 {
		if t1.Shape[0] != t2.Shape[0] {
			return nil, fmt.Errorf("vectors must have same length for dot product, got %d and %d", t1.Shape[0], t2.Shape[0])
		}

		result := 0.0
		for i := 0; i < t1.Shape[0]; i++ {
			result += t1.Data[i] * t2.Data[i]
		}
		return NewTensor(result), nil
	}

	// Case 2: Matrix dot Vector (2D * 1D)
	if len(t1.Shape) == 2 && len(t2.Shape) == 1 {
		if t1.Shape[1] != t2.Shape[0] {
			return nil, fmt.Errorf("matrix columns (%d) must match vector length (%d)", t1.Shape[1], t2.Shape[0])
		}

		result := make([]float64, t1.Shape[0])
		for i := 0; i < t1.Shape[0]; i++ {
			for j := 0; j < t1.Shape[1]; j++ {
				result[i] += t1.Data[i*t1.Shape[1]+j] * t2.Data[j]
			}
		}
		return NewTensor(result, t1.Shape[0]), nil
	}

	// Case 3: Matrix dot Matrix (2D * 2D)
	if len(t1.Shape) == 2 && len(t2.Shape) == 2 {
		if t1.Shape[1] != t2.Shape[0] {
			return nil, fmt.Errorf("matrix dimensions mismatch: %v and %v", t1.Shape, t2.Shape)
		}

		rows := t1.Shape[0]
		cols := t2.Shape[1]
		inner := t1.Shape[1]

		result := make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				sum := 0.0
				for k := 0; k < inner; k++ {
					sum += t1.Data[i*inner+k] * t2.Data[k*cols+j]
				}
				result[i*cols+j] = sum
			}
		}
		return NewTensor(result, rows, cols), nil
	}

	return nil, fmt.Errorf("unsupported tensor dimensions for dot product: %v and %v", t1.Shape, t2.Shape)
}

// SliceTime extracts a time slice from a sequence tensor
func SliceTime(t *Tensor, timeStep int) *Tensor {
	if len(t.Shape) < 3 {
		panic("tensor must have at least 3 dimensions for time slicing [batch, time, features]")
	}

	batchSize := t.Shape[0]
	seqLen := t.Shape[1]
	featureSize := t.Shape[2]

	result := make([]float64, batchSize*featureSize)

	// Copy data for each batch at the given time step
	for b := 0; b < batchSize; b++ {
		start := (b*seqLen + timeStep) * featureSize
		destStart := b * featureSize
		copy(result[destStart:destStart+featureSize], t.Data[start:start+featureSize])
	}

	return &Tensor{Data: result, Shape: []int{batchSize, featureSize}}
}

// Transpose swaps the dimensions of a tensor
func Transpose(t *Tensor) *Tensor {
	if len(t.Shape) != 2 {
		panic("transpose only implemented for 2D tensors")
	}
	rows, cols := t.Shape[0], t.Shape[1]
	result := make([]float64, len(t.Data))

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j*rows+i] = t.Data[i*cols+j]
		}
	}

	return &Tensor{Data: result, Shape: []int{cols, rows}}
}

// SumAxis sums a tensor along the specified axis
func SumAxis(t *Tensor, axis int) *Tensor {
	if axis >= len(t.Shape) {
		panic("axis out of bounds")
	}

	// For axis 0 of a 2D tensor, sum columns
	if axis == 0 && len(t.Shape) == 2 {
		result := make([]float64, t.Shape[1])
		for i := 0; i < t.Shape[0]; i++ {
			for j := 0; j < t.Shape[1]; j++ {
				result[j] += t.Data[i*t.Shape[1]+j]
			}
		}
		return &Tensor{Data: result, Shape: []int{t.Shape[1]}}
	}

	panic("SumAxis only implemented for axis 0 of 2D tensors")
}
