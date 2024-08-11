// Package model contains all of the core model training functions
package model

import (
	"fmt"
	lf "gotorch/loss_functions"
	"gotorch/tensor"
)

type Model interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(input, gradOutput *tensor.Tensor) *tensor.Tensor
}

type Linear struct {
	Weights     []float64
	Biases      []float64
	GradWeights []float64
	GradBiases  []float64
}

// Defines the forward propagation function
func (m *Linear) Forward(input *tensor.Tensor) *tensor.Tensor {

	if len(input.Shape) != 2 || input.Shape[1] != len(m.Weights) {
		panic(fmt.Sprintf("input shape %v is not compatible with weights size %d", input.Shape, len(m.Weights)))
	}

	// sets the batch size to the number of rows in the tensor
	batchSize := input.Shape[0]

	// prepare output tensor
	outputData := make([]float64, batchSize)
	for i := 0; i < batchSize; i++ { // for each row in the tensor
		for j, weight := range m.Weights { // iterate over the weights
			outputData[i] += weight * input.Data[i*len(m.Weights)+j] // adjust indexing for batch processing
		}
		outputData[i] += m.Biases[i%len(m.Biases)] // biasess are repeated for each batch
	}

	return tensor.NewTensor(outputData, batchSize, 1)
}

// Defines the backward propagation function
// input is the input tensor that is passed ot the forawrd function
// gradOutput is the gradient of the loss wrt the output of this layer, coming from the next layer in the network
func (m *Linear) Backward(input *tensor.Tensor, gradOutput *tensor.Tensor) *tensor.Tensor {

	if len(input.Shape) != 2 || input.Shape[1] != len(m.Weights) {
		panic(fmt.Sprintf("input shape %v is not compatible with weights size %d", input.Shape, len(m.Weights)))
	}

	// sets the batch size to the number of rows in the tensor
	batchSize := input.Shape[0]

	// initialize gradients with zeros
	m.GradWeights = make([]float64, len(m.Weights))
	m.GradBiases = make([]float64, len(m.Biases))
	gradInputData := make([]float64, len(input.Data)) // stores the gradient of the loss wrt input tensor

	for i := 0; i < batchSize; i++ { // for each row in the tensor
		for j := 0; j < len(m.Weights); j++ { // iterate over the weights
			m.GradWeights[j] += gradOutput.Data[i] * input.Data[i*len(m.Weights)+j] // calc the weight gradient
			gradInputData[i*len(m.Weights)+j] += gradOutput.Data[i] * m.Weights[j]  // calc grad input data
		}
		m.GradBiases[i%len(m.Biases)] += gradOutput.Data[i]
	}

	return tensor.NewTensor(gradInputData, batchSize, len(m.Weights))

}

// Defines the Train function which actually trains a model
func (m *Linear) Train(model *Linear, inputs, targets *tensor.Tensor, epochs int, learningRate float64) {
	optimizer := &SGD{LearningRate: learningRate}

	for epoch := 0; epoch < epochs; epoch++ {
		// forward pass
		predictions := model.Forward(inputs)

		// compute loss
		loss := lf.MSELoss(predictions, targets)

		// compute the gradient of the loss with respect to the output (gradOutput)
		gradOutputData := make([]float64, len(predictions.Data))
		for i := 0; i < len(predictions.Data); i++ {
			gradOutputData[i] = 2 * (predictions.Data[i] - targets.Data[i]) / float64(len(predictions.Data))
		}
		gradOutput := tensor.NewTensor(gradOutputData, predictions.Shape...)

		// backward pass
		model.Backward(inputs, gradOutput)

		// update weights
		optimizer.Step(model)

		if epoch%10 == 0 {
			fmt.Printf("Epoch %d: Loss = %f\n", epoch, loss)
		}
	}
}

// Defines a sample function which takes in a model and an input tensor and returns a new tensor
func (m *Linear) Sample(model *Linear, newInput *tensor.Tensor) *tensor.Tensor {
	return model.Forward(newInput)
}

type SGD struct {
	LearningRate float64
}

// implements stochastic gradient descent optimization function which updates the models weights and biases using the gradients computed during the backward pass
func (s *SGD) Step(model *Linear) {

	for i := range model.Weights {
		model.Weights[i] -= s.LearningRate * model.GradWeights[i]
	}

	for i := range model.Biases {
		model.Biases[i] -= s.LearningRate * model.GradBiases[i]
	}

}
