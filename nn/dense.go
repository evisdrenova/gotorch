// Package nn implements neural network layers and components
package nn

import (
	"gotorch/tensor"
	"math"
	"math/rand"
)

// Dense implements a fully connected neural network layer.
// It transforms input by applying weights and biases: output = input * weights + biases
type Dense struct {
	InputDim  int            // Number of input features
	OutputDim int            // Number of output features
	Weights   *tensor.Tensor // Weight matrix of shape [InputDim, OutputDim]
	Biases    *tensor.Tensor // Bias vector of shape [OutputDim]
	Input     *tensor.Tensor // Stores input for backpropagation

	// Gradients for optimization
	WeightGrads *tensor.Tensor // Accumulated gradients for weights
	BiasGrads   *tensor.Tensor // Accumulated gradients for biases
}

// NewDense creates a new dense layer with specified input and output dimensions.
// Weights are initialized using Xavier/Glorot initialization to help with training.
// Biases are initialized to zero.
func NewDense(inputDim, outputDim int) *Dense {
	// Xavier/Glorot initialization scale
	scale := math.Sqrt(2.0 / float64(inputDim+outputDim))

	// Initialize weights with scaled normal distribution
	weights := make([]float64, inputDim*outputDim)
	for i := range weights {
		weights[i] = rand.NormFloat64() * scale
	}

	// Initialize biases to zero
	biases := make([]float64, outputDim)

	return &Dense{
		InputDim:    inputDim,
		OutputDim:   outputDim,
		Weights:     tensor.NewTensor(weights, inputDim, outputDim),
		Biases:      tensor.NewTensor(biases, outputDim),
		WeightGrads: tensor.NewTensor(make([]float64, inputDim*outputDim), inputDim, outputDim),
		BiasGrads:   tensor.NewTensor(make([]float64, outputDim), outputDim),
	}
}

// Forward performs the forward pass computation.
// Given input x, computes output = x * W + b where:
// - x is the input tensor of shape [batch_size, input_dim]
// - W is the weight matrix of shape [input_dim, output_dim]
// - b is the bias vector of shape [output_dim]
func (d *Dense) Forward(input *tensor.Tensor) *tensor.Tensor {
	d.Input = input // Save for backprop

	// Compute output = input * weights + biases
	output, err := tensor.Dot(input, d.Weights)
	if err != nil {
		panic(err)
	}

	// Broadcast biases to match output shape
	batchSize := input.Shape[0]
	broadcastBiases := tensor.NewTensor(make([]float64, batchSize*d.OutputDim), batchSize, d.OutputDim)
	for i := 0; i < batchSize; i++ {
		copy(broadcastBiases.Data[i*d.OutputDim:(i+1)*d.OutputDim], d.Biases.Data)
	}

	result, err := tensor.Add(output, broadcastBiases)
	if err != nil {
		panic(err)
	}

	return result
}

// Backward performs the backward pass to compute gradients.
// Given gradient of loss with respect to output (gradOutput),
// computes:
// - Gradient with respect to input
// - Gradient with respect to weights
// - Gradient with respect to biases
func (d *Dense) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Compute gradients
	// dW = input^T * gradOutput
	inputT := tensor.Transpose(d.Input)
	d.WeightGrads, _ = tensor.Dot(inputT, gradOutput)

	// dB = sum(gradOutput, axis=0)
	d.BiasGrads = tensor.SumAxis(gradOutput, 0)

	// Compute gradient with respect to input
	// dX = gradOutput * W^T
	weightsT := tensor.Transpose(d.Weights)
	inputGrad, _ := tensor.Dot(gradOutput, weightsT)

	return inputGrad
}

// UpdateParams updates the layer parameters (weights and biases)
// using the computed gradients and the specified learning rate.
func (d *Dense) UpdateParams(learningRate float64) {
	// Update weights: W = W - lr * dW
	weightUpdate, _ := tensor.Multiply(d.WeightGrads, tensor.NewTensor(learningRate))
	d.Weights, _ = tensor.Subtract(d.Weights, weightUpdate)

	// Update biases: b = b - lr * db
	biasUpdate, _ := tensor.Multiply(d.BiasGrads, tensor.NewTensor(learningRate))
	d.Biases, _ = tensor.Subtract(d.Biases, biasUpdate)
}
