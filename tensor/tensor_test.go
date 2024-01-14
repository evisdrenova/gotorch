package tensor

import (
	"reflect"
	"testing"
)

/*
Unit tests for the Tensor package
*/

func Test_NewTensor(t *testing.T) {

	test := []float64{1.23, 4.56}

	tensor := NewTensor(test)

	if tensor.Data == nil {
		t.Errorf("the tensor cannot be nil")
	}

	if !reflect.DeepEqual(tensor.Data, test) {
		t.Errorf("expeced that the created tensor is a deep copy of test value")
	}
}

func Test_NewScalar(t *testing.T) {

	scalar := NewTensor(3.14)

	expectedShape := []int{1}
	if !reflect.DeepEqual(scalar.Shape, expectedShape) {
		t.Errorf("Incorrect shape for scalar: got %v, expected %v", scalar.Shape, expectedShape)
	}

	expectedData := []float64{3.14}
	if !reflect.DeepEqual(scalar.Data, expectedData) {
		t.Errorf("Incorrect data for scalar: got %v, expected %v", scalar.Data, expectedData)
	}

}

func Test_NewVector(t *testing.T) {

	vector := NewTensor([]float64{1, 2})

	expectedShape := []int{2}
	if !reflect.DeepEqual(vector.Shape, expectedShape) {
		t.Errorf("Incorrect shape for vector: got %v, expected %v", vector.Shape, expectedShape)
	}

	expectedData := []float64{1, 2}
	if !reflect.DeepEqual(vector.Data, expectedData) {
		t.Errorf("Incorrect data for vector: got %v, expected %v", vector.Data, expectedData)
	}

}

func Test_NewMatrix1x3(t *testing.T) {

	matrix := NewTensor([][]float64{{1, 2, 3}})

	// expect a 1x3 matrix
	expectedShape := []int{1, 3}
	if !reflect.DeepEqual(matrix.Shape, expectedShape) {
		t.Errorf("Incorrect shape for matrix: got %v, expected %v", matrix.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3}
	if !reflect.DeepEqual(matrix.Data, expectedData) {
		t.Errorf("Incorrect data for matrix: got %v, expected %v", matrix.Data, expectedData)
	}
}

func Test_NewMatrix2x3(t *testing.T) {

	matrix := NewTensor([][]float64{{1, 2, 3}, {4, 5, 6}})

	// expect a 2x3 matrix
	expectedShape := []int{2, 3}
	if !reflect.DeepEqual(matrix.Shape, expectedShape) {
		t.Errorf("Incorrect shape for matrix: got %v, expected %v", matrix.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(matrix.Data, expectedData) {
		t.Errorf("Incorrect data for matrix: got %v, expected %v", matrix.Data, expectedData)
	}
}

func Test_NewMatrix4x2(t *testing.T) {

	matrix := NewTensor([][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}})

	// expect a 2x3 matrix
	expectedShape := []int{4, 2}
	if !reflect.DeepEqual(matrix.Shape, expectedShape) {
		t.Errorf("Incorrect shape for matrix: got %v, expected %v", matrix.Shape, expectedShape)
	}

	// since we flatten the data, we expect a 1-dimensional slice
	expectedData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	if !reflect.DeepEqual(matrix.Data, expectedData) {
		t.Errorf("Incorrect data for matrix: got %v, expected %v", matrix.Data, expectedData)
	}
}

func TestDimsScalar(t *testing.T) {

	scalar := NewTensor(3)

	expected := 1

	if scalar.Dims() != expected {
		t.Errorf("Incorrect dimensions for scalar, got %d, expected: %d", scalar.Dims(), expected)
	}
}

func TestDimsVector(t *testing.T) {

	vector := NewTensor([]float64{1, 2})

	expected := 1

	if vector.Dims() != expected {
		t.Errorf("Incorrect dimensions for vector, got %d, expected: %d", vector.Dims(), expected)
	}
}

func TestDimsMatrix(t *testing.T) {

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

func Test_AddTenso1D(t *testing.T) {
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

func Test_AddTensor1D(t *testing.T) {
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

func Test_AddTensor2D(t *testing.T) {
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
