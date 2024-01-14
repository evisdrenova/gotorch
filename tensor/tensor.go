package tensor

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
// spreads the shape param to allow specifying different shaped tensors
func NewTensor(data []float64, shape ...int) *Tensor {

	if len(data) < 1 {
		return nil
	}

	numElements := 1
	for _, dim := range shape {
		numElements *= dim
	}

	if numElements != len(data) {
		return nil
	}

	return &Tensor{Data: data, Shape: shape}
}

// creates a new scalar value
func NewScalar(value float64) *Tensor {
	return NewTensor([]float64{value}, 1)
}

// creates a new vector value
func NewVector(value []float64) *Tensor {
	return NewTensor(value, len(value))
}

// creates a new matrix value
func NewMatrix(data [][]float64) *Tensor {

	flatData := make([]float64, 0, len(data)*len(data[0]))

	// we flatten the matrix into a 1-dimensional slice to make it easy and fast to work with later
	// we maintain the matrix shape in the t.Shape field so we can reconstruct the shape if necessary
	for _, row := range data {
		flatData = append(flatData, row...)
	}

	return NewTensor(flatData, len(data), len(data[0]))
}

// returns the number of dimensions for a tensor
func (t *Tensor) Dims() int {
	return len(t.Shape)
}
