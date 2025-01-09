package nn

import (
	"gotorch/tensor"
	"reflect"
	"testing"
)

func Test_NewDense(t *testing.T) {
	inputDim := 4
	outputDim := 3

	dense := NewDense(inputDim, outputDim)

	// Check dimensions
	expectedWeightShape := []int{inputDim, outputDim}
	if !reflect.DeepEqual(dense.Weights.Shape, expectedWeightShape) {
		t.Errorf("Incorrect weight shape: got %v, want %v", dense.Weights.Shape, expectedWeightShape)
	}

	expectedBiasShape := []int{outputDim}
	if !reflect.DeepEqual(dense.Biases.Shape, expectedBiasShape) {
		t.Errorf("Incorrect bias shape: got %v, want %v", dense.Biases.Shape, expectedBiasShape)
	}
}

func Test_DenseForward(t *testing.T) {
	dense := NewDense(2, 3)

	// Set weights and biases for predictable output
	weights := tensor.NewTensor([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	biases := tensor.NewTensor([]float64{0.1, 0.2, 0.3})

	dense.Weights = weights
	dense.Biases = biases
	dense.WeightGrads = tensor.NewTensor(make([]float64, 2*3), 2, 3)
	dense.BiasGrads = tensor.NewTensor(make([]float64, 3), 3)

	// Input: batch size of 2, each with 2 features
	input := tensor.NewTensor([][]float64{
		{1, 2},
		{3, 4},
	})

	output := dense.Forward(input)

	expectedShape := []int{2, 3}
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	// Expected output calculation:
	// First sample: [1*1 + 2*4 + 0.1, 1*2 + 2*5 + 0.2, 1*3 + 2*6 + 0.3]
	// Second sample: [3*1 + 4*4 + 0.1, 3*2 + 4*5 + 0.2, 3*3 + 4*6 + 0.3]
	expectedOutput := []float64{
		9.1, 12.2, 15.3, // First sample
		19.1, 26.2, 33.3, // Second sample
	}

	for i, val := range output.Data {
		if !floatEquals(val, expectedOutput[i], 1e-6) {
			t.Errorf("Incorrect output at index %d: got %v, want %v", i, val, expectedOutput[i])
		}
	}
}

func Test_DenseBackward(t *testing.T) {
	dense := NewDense(2, 3)

	// Set weights for predictable output
	dense.Weights = tensor.NewTensor([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	// Forward pass to set input
	input := tensor.NewTensor([][]float64{
		{1, 2},
		{3, 4},
	})
	dense.Forward(input)

	// Backward pass
	gradOutput := tensor.NewTensor([][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	})

	inputGrad := dense.Backward(gradOutput)

	// Check input gradient shape
	expectedInputGradShape := []int{2, 2}
	if !reflect.DeepEqual(inputGrad.Shape, expectedInputGradShape) {
		t.Errorf("Incorrect input gradient shape: got %v, want %v", inputGrad.Shape, expectedInputGradShape)
	}

	// Check weight gradient shape
	expectedWeightGradShape := []int{2, 3}
	if !reflect.DeepEqual(dense.WeightGrads.Shape, expectedWeightGradShape) {
		t.Errorf("Incorrect weight gradient shape: got %v, want %v", dense.WeightGrads.Shape, expectedWeightGradShape)
	}
}
