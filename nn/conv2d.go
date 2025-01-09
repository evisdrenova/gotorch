// Package nn implements neural network layers and components
package nn

import (
	"gotorch/tensor"
	"math"
	"math/rand"
)

// Conv2D implements a 2D convolutional layer.
// Applies sliding filters over a 4D input tensor [batch_size, channels, height, width]
// to produce feature maps through local connectivity patterns.
type Conv2D struct {
	InChannels  int // Number of input channels
	OutChannels int // Number of output channels (filters)
	KernelSize  int // Size of the convolving kernel (assumed square)
	Stride      int // Step size of the convolution
	Padding     int // Zero-padding size around the input

	Filters *tensor.Tensor // Filter weights of shape [out_channels, in_channels, kernel_size, kernel_size]
	Biases  *tensor.Tensor // Bias terms of shape [out_channels]
	Input   *tensor.Tensor // Stores input for backpropagation

	// Gradients for optimization
	FilterGrads *tensor.Tensor // Accumulated gradients for filters
	BiasGrads   *tensor.Tensor // Accumulated gradients for biases
}

// NewConv2D creates a new 2D convolutional layer with specified dimensions.
// Filters are initialized using He/Kaiming initialization, which is particularly
// suitable for layers using ReLU activation.
func NewConv2D(inChannels, outChannels, kernelSize int, stride, padding int) *Conv2D {
	// He/Kaiming initialization scale
	scale := math.Sqrt(2.0 / float64(inChannels*kernelSize*kernelSize))

	// Initialize filters with scaled normal distribution
	filters := make([]float64, outChannels*inChannels*kernelSize*kernelSize)
	for i := range filters {
		filters[i] = rand.NormFloat64() * scale
	}

	// Initialize biases to zero
	biases := make([]float64, outChannels)

	return &Conv2D{
		InChannels:  inChannels,
		OutChannels: outChannels,
		KernelSize:  kernelSize,
		Stride:      stride,
		Padding:     padding,
		Filters:     tensor.NewTensor(filters, outChannels, inChannels, kernelSize, kernelSize),
		Biases:      tensor.NewTensor(biases, outChannels),
	}
}

// Forward performs the forward pass of the convolution operation.
// For each filter:
// 1. Applies filter across input volume in a sliding window fashion
// 2. Computes dot product between filter and input window at each position
// 3. Adds bias term to create output feature map
func (c *Conv2D) Forward(input *tensor.Tensor) *tensor.Tensor {
	c.Input = input // Save for backprop

	batchSize := input.Shape[0]
	inputHeight := input.Shape[2]
	inputWidth := input.Shape[3]

	// Calculate output dimensions with padding and stride
	outputHeight := (inputHeight+2*c.Padding-c.KernelSize)/c.Stride + 1
	outputWidth := (inputWidth+2*c.Padding-c.KernelSize)/c.Stride + 1

	// Create output tensor
	output := tensor.NewTensor(
		make([]float64, batchSize*c.OutChannels*outputHeight*outputWidth),
		batchSize, c.OutChannels, outputHeight, outputWidth,
	)

	// TODO: Implement convolution operation using im2col and matrix multiplication
	// Current implementation is a placeholder that copies input to output
	copy(output.Data, input.Data[:len(output.Data)])

	return output
}

// TODO: Add Backward and UpdateParams methods
// Backward should compute:
// - Gradient with respect to input
// - Gradient with respect to filters
// - Gradient with respect to biases
//
// UpdateParams should update filters and biases using computed gradients
