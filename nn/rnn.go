// Package nn implements neural network layers and components
package nn

import (
	af "gotorch/activation_functions"
	"gotorch/tensor"
	"math"
	"math/rand"
)

// RNN implements a simple Recurrent Neural Network layer.
// Processes sequences by maintaining a hidden state that is updated at each time step:
// h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
type RNN struct {
	InputDim  int // Size of input features at each time step
	HiddenDim int // Size of hidden state

	Wxh *tensor.Tensor // Input-to-hidden weights [input_dim, hidden_dim]
	Whh *tensor.Tensor // Hidden-to-hidden weights [hidden_dim, hidden_dim]
	Bh  *tensor.Tensor // Hidden bias [hidden_dim]

	States []*tensor.Tensor // Stores all hidden states for backpropagation [seq_len+1, batch_size, hidden_dim]
	Inputs []*tensor.Tensor // Stores inputs for backpropagation [seq_len, batch_size, input_dim]
}

// NewRNN creates a new RNN layer with specified input and hidden dimensions.
// Weights are initialized using Xavier/Glorot initialization to help with training.
// The hidden-to-hidden matrix (Whh) is carefully initialized to avoid vanishing/exploding gradients.
func NewRNN(inputDim, hiddenDim int) *RNN {
	// Xavier/Glorot initialization scale
	scale := math.Sqrt(2.0 / float64(inputDim+hiddenDim))

	// Initialize input-to-hidden weights
	wxh := make([]float64, inputDim*hiddenDim)
	for i := range wxh {
		wxh[i] = rand.NormFloat64() * scale
	}

	// Initialize hidden-to-hidden weights
	whh := make([]float64, hiddenDim*hiddenDim)
	for i := range whh {
		whh[i] = rand.NormFloat64() * scale
	}

	// Initialize biases to zero
	bh := make([]float64, hiddenDim)

	return &RNN{
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
		Wxh:       tensor.NewTensor(wxh, inputDim, hiddenDim),
		Whh:       tensor.NewTensor(whh, hiddenDim, hiddenDim),
		Bh:        tensor.NewTensor(bh, hiddenDim),
	}
}

// Forward performs the forward pass of the RNN.
// Input shape: [batch_size, seq_len, input_dim]
// For each time step t:
// 1. Extract input slice x_t
// 2. Compute linear transformations W_xh * x_t and W_hh * h_{t-1}
// 3. Add bias and apply tanh activation
// Returns the final hidden state
func (r *RNN) Forward(input *tensor.Tensor) *tensor.Tensor {
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]

	// Initialize storage for states and inputs
	r.States = make([]*tensor.Tensor, seqLen+1)
	r.Inputs = make([]*tensor.Tensor, seqLen)

	// Initialize first hidden state to zeros
	r.States[0] = tensor.NewTensor([][]float64{
		make([]float64, r.HiddenDim),
	}, batchSize, r.HiddenDim)

	// Process each time step
	for t := 0; t < seqLen; t++ {
		// Store input for backprop
		r.Inputs[t] = tensor.SliceTime(input, t)

		// Calculate hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
		wxh_x, err := tensor.Dot(r.Inputs[t], r.Wxh)
		if err != nil {
			panic(err)
		}
		whh_h, err := tensor.Dot(r.States[t], r.Whh)
		if err != nil {
			panic(err)
		}
		sum, err := tensor.Add(wxh_x, whh_h)
		if err != nil {
			panic(err)
		}

		// FIXME: Current bias broadcasting is a workaround
		// Need proper tensor broadcasting support
		broadcastBias := tensor.NewTensor(make([]float64, batchSize*r.HiddenDim), batchSize, r.HiddenDim)
		for i := 0; i < batchSize; i++ {
			copy(broadcastBias.Data[i*r.HiddenDim:(i+1)*r.HiddenDim], r.Bh.Data)
		}

		sum, err = tensor.Add(sum, broadcastBias)
		if err != nil {
			panic(err)
		}
		r.States[t+1] = af.Tanh(sum)
	}

	return r.States[len(r.States)-1]
}

// TODO: Add Backward and UpdateParams methods
// Backward should implement truncated backpropagation through time (TBPTT)
// to compute gradients for Wxh, Whh, and Bh
//
// UpdateParams should update all parameters using computed gradients

// TODO: Known Issues
// 1. Forward pass has shape mismatch when adding bias to intermediate states
// 2. Need to implement proper broadcasting in tensor.Add for bias addition
// 3. Consider adding a dedicated broadcasting function in tensor package
// 4. Current workaround uses manual broadcasting which may not be optimal
