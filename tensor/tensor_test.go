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

	scalar := NewScalar(3.14)

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

	vector := NewVector([]float64{1, 2})

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

	matrix := NewMatrix([][]float64{{1, 2, 3}})

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

	matrix := NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})

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

	matrix := NewMatrix([][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}})

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
