package model

import (
	"fmt"
	"gotorch/tensor"
	"testing"
)

func TestLinearForward1x2Matrix(t *testing.T) {
	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}})
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	expectedOutput := []float64{
		1.0*weights[0] + 2.0*weights[1] + biases[0], // Output for first batch row
	}

	output := model.Forward(inputTensor)

	fmt.Println("output", output)

	for i := range expectedOutput {
		if output.Data[i] != expectedOutput[i] {
			t.Errorf("Expected output %v, got %v at index %d", expectedOutput[i], output.Data[i], i)
		}
	}
}

func TestLinearForward2x2Matrix(t *testing.T) {
	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	expectedOutput := []float64{
		1.0*weights[0] + 2.0*weights[1] + biases[0], // Output for first batch row
		3.0*weights[0] + 4.0*weights[1] + biases[0], // Output for second batch row
	}

	output := model.Forward(inputTensor)

	for i := range expectedOutput {
		if output.Data[i] != expectedOutput[i] {
			t.Errorf("Expected output %v, got %v at index %d", expectedOutput[i], output.Data[i], i)
		}
	}
}

func TestLinearForward2x3Matrix(t *testing.T) {
	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}})
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	expectedOutput := []float64{
		1.0*weights[0] + 2.0*weights[1] + biases[0], // Output for first batch row
		3.0*weights[0] + 4.0*weights[1] + biases[0], // Output for second batch row
		5.0*weights[0] + 6.0*weights[1] + biases[0], // Output for third batch row
	}

	output := model.Forward(inputTensor)

	for i := range expectedOutput {
		if output.Data[i] != expectedOutput[i] {
			t.Errorf("Expected output %v, got %v at index %d", expectedOutput[i], output.Data[i], i)
		}
	}
}

func TestLinearBackward1x2Matrix(t *testing.T) {

	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	gradOutput := tensor.NewTensor([]float64{1.0, 2.0}, 2, 1) // Example gradient from next layer
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	gradInput := model.Backward(inputTensor, gradOutput)

	expectedGradWeights := []float64{7.0, 10.0}        // Expected values
	expectedGradBiases := []float64{3.0}               // Expected value
	expectedGradInput := []float64{1.0, 2.0, 2.0, 4.0} // Expected value

	// Compare the results
	for i, grad := range model.GradWeights {
		if grad != expectedGradWeights[i] {
			t.Errorf("Expected gradWeight %v, got %v", expectedGradWeights[i], grad)
		}
	}

	for i, grad := range model.GradBiases {
		if grad != expectedGradBiases[i] {
			t.Errorf("Expected gradBias %v, got %v", expectedGradBiases[i], grad)
		}
	}

	for i, grad := range gradInput.Data {
		if grad != expectedGradInput[i] {
			t.Errorf("Expected gradInput %v, got %v", expectedGradInput[i], grad)
		}
	}

}

// func TestFitFunction(t *testing.T) {
// 	// Create a simple model and data for testing
// 	model := &Linear{Weights: []float64{1.0, 2.0}, Biases: []float64{0.5}}
// 	data := tensor.NewTensor([]float64{3, 4, 5, 6}, 2, 2)
// 	labels := tensor.NewTensor([]float64{19.5, 24.5}, 2)

// 	// Run fit function for a single epoch
// 	Fit(model, data, labels, 1, 0.01)

// 	// Check if weights and biases are updated (test case can be more elaborate)
// 	if model.Weights[0] == 1.0 && model.Weights[1] == 2.0 && model.Biases[0] == 0.5 {
// 		t.Errorf("Fit function did not update model parameters")
// 	}
// }
