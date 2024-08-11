package model

import (
	"fmt"
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

// implements Fit function to train the model
// func Fit(model *Linear, data *tensor.Tensor, labels *tensor.Tensor, epochs int, learningRate float64) {
// 	batchSize := data.Shape[0]
// 	numFeatures := len(model.Weights)

// 	for epoch := 0; epoch < epochs; epoch++ {

// 		totalLoss := 0.0

// 		for i := 0; i < batchSize; i++ {
// 			// get a single batch
// 			inputBatch := tensor.NewTensor(data.Data[i*numFeatures:(i+1)*numFeatures], 1, numFeatures)

// 			output := model.Forward(inputBatch)
// 			expected := tensor.NewTensor([]float64{labels.Data[i]}, 1)

// 			// calc loss
// 			lossTensor := nn.MSELoss(output, expected)
// 			loss := lossTensor.Data[0]
// 			totalLoss += loss

// 			// gradient calculation for weights - replace this with SGD or something else
// 			for j := range model.Weights {
// 				model.Weights[j] -= learningRate * 2 * (output.Data[0] - expected.Data[0]) * inputBatch.Data[j]
// 			}

// 			// gradient calculation for biases - replace this with SGD or something else
// 			for j := range model.Biases {
// 				model.Biases[j] -= learningRate * 2 * (output.Data[0] - expected.Data[0])
// 			}
// 		}

// 		avgLoss := totalLoss / float64(batchSize)
// 		fmt.Printf("Epoch [%d/%d], Loss: %.4f\n", epoch+1, epochs, avgLoss)
// 	}
// }
