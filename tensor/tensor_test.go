package tensor

import (
	"reflect"
	"testing"
)

/*
Unit tests for the Tensor package
*/

func Test_NewTensorScalar(t *testing.T) {

	tensor := NewTensor(3.14)

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.14}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}

}

func Test_NewTensorVector(t *testing.T) {

	tensor := NewTensor([]float64{1, 2})

	expectedShape := []int{2}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{1, 2}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for vector: got %v, expected %v", tensor.Data, expectedData)
	}

}

func Test_NewTensorMatrix1x3(t *testing.T) {

	tensor := NewTensor([][]float64{{1, 2, 3}})

	// expect a 1x3 matrix
	expectedShape := []int{1, 3}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorMatrix2x3(t *testing.T) {

	tensor := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})

	// expect a 2x3 matrix
	expectedShape := []int{2, 3}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorMatrix4x2(t *testing.T) {

	tensor := NewTensor([][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}})

	// expect a 2x3 matrix
	expectedShape := []int{4, 2}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorInt(t *testing.T) {
	tensor := NewTensor(3)

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorInt32(t *testing.T) {
	tensor := NewTensor(int32(3))

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorInt64(t *testing.T) {
	tensor := NewTensor(int64(3))

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorFloat64(t *testing.T) {
	tensor := NewTensor(float64(3.53))

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.53}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorFloat64Slice(t *testing.T) {
	tensor := NewTensor([]float64{3.53})

	expectedShape := []int{1}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.53}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorFloat64SliceSlice(t *testing.T) {
	tensor := NewTensor([][]float64{{3.53, 234.2}})

	expectedShape := []int{1, 2}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	expectedData := []float64{3.53, 234.2}
	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data for tensor: got %v, expected %v", tensor.Data, expectedData)
	}
}

func Test_NewTensorPanic(t *testing.T) {

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	_ = NewTensor("hello")

}

func Test_NewTensorFrom2DSliceZeroLength(t *testing.T) {
	tensor := newTensorFrom2DSlice([][]float64{})

	expectedShape := []int{0, 0}
	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape for tensor: got %v, expected %v", tensor.Shape, expectedShape)
	}

	if len(tensor.Data) != 0 {
		t.Errorf("Data slice should be empty, got %v", tensor.Data)
	}

}

func Test_NewTensorFRom2DslicePanic(t *testing.T) {

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic for rows of different lengths")
		}
	}()

	// This should cause a panic due to rows of different lengths
	_ = newTensorFrom2DSlice([][]float64{{1, 2}, {3}})

}

func TestDimsTensorScalar(t *testing.T) {

	scalar := NewTensor(3)

	expected := 1

	if scalar.Dims() != expected {
		t.Errorf("Incorrect dimensions for scalar, got %d, expected: %d", scalar.Dims(), expected)
	}
}

func TestDimsTensorVector(t *testing.T) {

	vector := NewTensor([]float64{1, 2})

	expected := 1

	if vector.Dims() != expected {
		t.Errorf("Incorrect dimensions for vector, got %d, expected: %d", vector.Dims(), expected)
	}
}

func TestDimsTensorMatrix(t *testing.T) {

	matrix := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})

	expected := 2

	if matrix.Dims() != expected {
		t.Errorf("Incorrect dimensions for matrix, got %d, expected: %d", matrix.Dims(), expected)
	}
}

func Test_AddError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Add(s1, s2)
	if err == nil {
		t.Errorf("Cannot add two tensors of different shapes and dimensions")
	}
}

func Test_AddTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Add(s1, s2)
	if err != nil {
		t.Errorf("unable to add two tensor")
	}

	expected := float64(6)
	if result.Data[0] != expected {
		t.Errorf("Unable to add scalars correctly")
	}

}

func Test_AddTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{6, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Add(s1, s2)
	if err != nil {
		t.Errorf("unable to add two tensor")
	}

	expected := []float64{10, 4}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect addition,expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_AddTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{6, 3}, {2, 6}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Add(s1, s2)
	if err != nil {
		t.Errorf("unable to add two tensor")
	}

	expected := []float64{10, 4, 7, 15} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect addition at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func Test_SubtractError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Subtract(s1, s2)
	if err == nil {
		t.Errorf("Cannot subtract two tensors of different shapes and dimensions")
	}
}

func Test_SubtractTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Subtract(s1, s2)
	if err != nil {
		t.Errorf("unable to subtract two tensor")
	}

	expected := float64(-2)
	if result.Data[0] != expected {
		t.Errorf("Unable to subtract scalars correctly, expected: %v, got: %v", expected, result.Data[0])
	}

}

func Test_SubtractTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{6, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Subtract(s1, s2)
	if err != nil {
		t.Errorf("unable to subtract two tensor")
	}

	expected := []float64{2, 2}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect subtraction, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_SubtractTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{6, 3}, {2, 6}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Subtract(s1, s2)
	if err != nil {
		t.Errorf("unable to subtract two tensor")
	}

	expected := []float64{2, 2, -3, -3} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect subtraction at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func Test_MultiplyError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Multiply(s1, s2)
	if err == nil {
		t.Errorf("Cannot multiply two tensors of different shapes and dimensions")
	}
}

func Test_MultiplyTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Multiply(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := float64(8)
	if result.Data[0] != expected {
		t.Errorf("Unable to multiply scalars correctly, expected: %v, got: %v", expected, result.Data[0])
	}

}

func Test_MultiplyTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{6, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Multiply(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{24, 3}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect multiplication, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_MultiplyTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{6, 3}, {2, 6}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Multiply(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{24, 3, 10, 54} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect multiplication at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}

func TestDivideError(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor([]float64{1, 2})

	_, err := Divide(s1, s2)
	if err == nil {
		t.Errorf("Cannot divide two tensors of different shapes and dimensions")
	}
}

func Test_DivideTensorScalar(t *testing.T) {
	s1 := NewTensor(2)
	s2 := NewTensor(4)

	result, err := Divide(s1, s2)
	if err != nil {
		t.Errorf("unable to divide two tensor")
	}

	expected := float64(0.5)
	if result.Data[0] != expected {
		t.Errorf("Unable to divide scalars correctly, expected: %v, got: %v", expected, result.Data[0])
	}

}

func Test_DivideTensorVector(t *testing.T) {
	s1 := NewTensor([]float64{8, 3})
	s2 := NewTensor([]float64{4, 1})

	result, err := Divide(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{2, 3}
	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("incorrect division, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_DivideTensorMatrix(t *testing.T) {
	s1 := NewTensor([][]float64{{8, 3}, {10, 27}})
	s2 := NewTensor([][]float64{{4, 1}, {5, 9}})

	result, err := Divide(s1, s2)
	if err != nil {
		t.Errorf("unable to multiply two tensor")
	}

	expected := []float64{2, 3, 2, 3} // Flattened 2x2 matrix
	if len(result.Data) != len(expected) {
		t.Errorf("resulting tensor has incorrect number of elements: got %v, want %v", len(result.Data), len(expected))
	}

	for i, v := range result.Data {
		if v != expected[i] {
			t.Errorf("incorrect division at index %d, expected: %v, got: %v", i, expected[i], v)
		}
	}
}
