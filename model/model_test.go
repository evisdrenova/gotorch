package model

import (
	"fmt"
	"gotorch/tensor"
	"testing"
)

func TestLinearForward1x2Tensor(t *testing.T) {
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

func TestLinearForward2x2Tensor(t *testing.T) {
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

func TestLinearForward3x2Tensor(t *testing.T) {
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

func TestLinearBackward1x2Tensor(t *testing.T) {

	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}})
	gradOutput := tensor.NewTensor([]float64{1.0, 2.0}, 1, 1) // Example gradient from next layer
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	gradInput := model.Backward(inputTensor, gradOutput)

	expectedGradWeights := []float64{1.0, 2.0}
	expectedGradBiases := []float64{1.0}
	expectedGradInput := []float64{1.0, 2.0}

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

func TestLinearBackward2x2Tensor(t *testing.T) {

	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	gradOutput := tensor.NewTensor([]float64{1.0, 2.0}, 2, 1) // Example gradient from next layer
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	gradInput := model.Backward(inputTensor, gradOutput)

	//∑(gradOutput[i]×inputTensor[i,j])
	// so batch 1 = (1.0×1.0)+(2.0×3.0)=1.0+6.0=7.0
	// and  bacth 2 = (1.0×2.0)+(2.0×4.0)=2.0+8.0=10.0
	expectedGradWeights := []float64{7.0, 10.0}
	expectedGradBiases := []float64{3.0}
	expectedGradInput := []float64{1.0, 2.0, 2.0, 4.0}

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

func TestLinearBackward3x2Tensor(t *testing.T) {

	inputTensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}})
	gradOutput := tensor.NewTensor([]float64{1.0, 2.0, 3.0}, 3, 1) // Example gradient from next layer
	weights := []float64{1.0, 2.0}
	biases := []float64{0.5}
	model := &Linear{Weights: weights, Biases: biases}

	gradInput := model.Backward(inputTensor, gradOutput)

	expectedGradWeights := []float64{22.0, 28.0}
	expectedGradBiases := []float64{6.0}
	expectedGradInput := []float64{1.0, 2.0, 2.0, 4.0, 3.0, 6.0}

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

func TestSGDStep(t *testing.T) {
	initialWeights := []float64{0.5, -1.5}
	initialBiases := []float64{0.0}
	gradWeights := []float64{0.1, -0.2}
	gradBiases := []float64{0.05}

	model := &Linear{
		Weights:     initialWeights,
		Biases:      initialBiases,
		GradWeights: gradWeights,
		GradBiases:  gradBiases,
	}

	optimizer := &SGD{LearningRate: 0.1}

	expectedWeights := []float64{0.5 - 0.1*0.1, -1.5 - 0.1*(-0.2)} // {0.49, -1.48}
	expectedBiases := []float64{0.0 - 0.1*0.05}                    // {-0.005}

	optimizer.Step(model)

	// Check if the weights are updated correctly
	for i, weight := range model.Weights {
		if weight != expectedWeights[i] {
			t.Errorf("Weight %d: expected %f, got %f", i, expectedWeights[i], weight)
		}
	}

	// Check if the biases are updated correctly
	if model.Biases[0] != expectedBiases[0] {
		t.Errorf("Bias: expected %f, got %f", expectedBiases[0], model.Biases[0])
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
