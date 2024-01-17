package model

import (
	"gotorch/tensor"
	"testing"
)

func TestLinearForward(t *testing.T) {
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	inputData := tensor.NewTensor([]float64{3, 4}, 1, 2)
	expectedOutput := 3*weights[0] + 4*weights[1] + biases[0]

	output := model.Forward(inputData)

	if output.Data[0] != expectedOutput {
		t.Errorf("Expected output %v, got %v", expectedOutput, output.Data[0])
	}
}

func TestFitFunction(t *testing.T) {
	// Create a simple model and data for testing
	model := &Linear{Weights: []float64{1.0, 2.0}, Biases: []float64{0.5}}
	data := tensor.NewTensor([]float64{3, 4, 5, 6}, 2, 2)
	labels := tensor.NewTensor([]float64{19.5, 24.5}, 2)

	// Run fit function for a single epoch
	Fit(model, data, labels, 1, 0.01)

	// Check if weights and biases are updated (test case can be more elaborate)
	if model.Weights[0] == 1.0 && model.Weights[1] == 2.0 && model.Biases[0] == 0.5 {
		t.Errorf("Fit function did not update model parameters")
	}
}
