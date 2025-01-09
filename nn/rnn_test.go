package nn

import (
	"gotorch/tensor"
	"math"
	"reflect"
	"testing"
)

func Test_NewRNN(t *testing.T) {
	inputDim := 4
	hiddenDim := 3

	rnn := NewRNN(inputDim, hiddenDim)

	// Check dimensions
	expectedWxhShape := []int{inputDim, hiddenDim}
	if !reflect.DeepEqual(rnn.Wxh.Shape, expectedWxhShape) {
		t.Errorf("Incorrect Wxh shape: got %v, want %v", rnn.Wxh.Shape, expectedWxhShape)
	}

	expectedWhhShape := []int{hiddenDim, hiddenDim}
	if !reflect.DeepEqual(rnn.Whh.Shape, expectedWhhShape) {
		t.Errorf("Incorrect Whh shape: got %v, want %v", rnn.Whh.Shape, expectedWhhShape)
	}

	expectedBhShape := []int{hiddenDim}
	if !reflect.DeepEqual(rnn.Bh.Shape, expectedBhShape) {
		t.Errorf("Incorrect bias shape: got %v, want %v", rnn.Bh.Shape, expectedBhShape)
	}
}

// TODO: Known Issues
// 1. Test_RNNForward fails due to shape mismatch in bias addition
// 2. Current test uses simplified bias handling that doesn't match implementation
// 3. Need to add tests for:
//    - Proper bias broadcasting
//    - Gradient computation
//    - Parameter updates
//    - More complex sequences

// test fails due to tensor shape mismatch in bias addition
func Test_RNNForward(t *testing.T) {
	inputDim := 2
	hiddenDim := 3
	rnn := NewRNN(inputDim, hiddenDim)

	// Set weights for predictable output
	rnn.Wxh = tensor.NewTensor([][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	})
	rnn.Whh = tensor.NewTensor([][]float64{
		{0.7, 0.8, 0.9},
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	})
	// Bias should be a vector [hidden_dim]
	rnn.Bh = tensor.NewTensor([]float64{0.1, 0.2, 0.3}, hiddenDim)

	// Input: batch size of 2, sequence length of 2, input dim of 2
	input := tensor.NewTensor([]float64{
		0.1, 0.2, // batch 0, time 0
		0.3, 0.4, // batch 0, time 1
		0.5, 0.6, // batch 1, time 0
		0.7, 0.8, // batch 1, time 1
	}, 2, 2, 2) // [batch_size, seq_len, input_dim]

	output := rnn.Forward(input)

	// Check output dimensions
	expectedShape := []int{2, hiddenDim} // [batch_size, hidden_dim]
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	// Check states were stored correctly
	if len(rnn.States) != 3 { // seqLen + 1
		t.Errorf("Incorrect number of states stored: got %d, want %d", len(rnn.States), 3)
	}

	if len(rnn.Inputs) != 2 { // seqLen
		t.Errorf("Incorrect number of inputs stored: got %d, want %d", len(rnn.Inputs), 2)
	}
}

// test passes but uses simplified bias handling
func Test_RNNForwardSingleTimeStep(t *testing.T) {
	inputDim := 2
	hiddenDim := 2
	rnn := NewRNN(inputDim, hiddenDim)

	// Set simple identity weights
	rnn.Wxh = tensor.NewTensor([][]float64{
		{1, 0},
		{0, 1},
	})
	rnn.Whh = tensor.NewTensor([][]float64{
		{0, 0},
		{0, 0},
	})
	// Bias needs to match output shape [batch_size, hidden_dim]
	rnn.Bh = tensor.NewTensor([]float64{0, 0}, hiddenDim)

	// Single time step input with shape [batch_size=1, seq_len=1, input_dim=2]
	input := tensor.NewTensor([]float64{1, 2}, 1, 1, 2)

	output := rnn.Forward(input)

	expectedShape := []int{1, hiddenDim}
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	// With identity weights and zero bias, output should be tanh of input
	expectedFirstValue := math.Tanh(1.0)
	expectedSecondValue := math.Tanh(2.0)
	if !floatEquals(output.Data[0], expectedFirstValue, 1e-6) {
		t.Errorf("Incorrect first output value: got %v, want %v", output.Data[0], expectedFirstValue)
	}
	if !floatEquals(output.Data[1], expectedSecondValue, 1e-6) {
		t.Errorf("Incorrect second output value: got %v, want %v", output.Data[1], expectedSecondValue)
	}
}
