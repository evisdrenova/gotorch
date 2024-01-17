package model

import (
	"fmt"
	nn "gotorch/neuralnet"
	"gotorch/tensor"
)

type Model interface {
	Forward(input *tensor.Tensor) tensor.Tensor
}

type Linear struct {
	Weights []float64
	Biases  []float64
}

func (m *Linear) Forward(input *tensor.Tensor) *tensor.Tensor {

	if len(input.Shape) != 2 || input.Shape[1] != len(m.Weights) {
		panic(fmt.Sprintf("input shape %v is not compatible with weights size %d", input.Shape, len(m.Weights)))
	}

	batchSize := input.Shape[0]

	// Prepare output tensor
	outputData := make([]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		for j, weight := range m.Weights {
			outputData[i] += weight * input.Data[i*len(m.Weights)+j] // Adjust indexing for batch processing
		}
		outputData[i] += m.Biases[i%len(m.Biases)] // Handling biases (assuming biases are repeated for each input in the batch)
	}

	return tensor.NewTensor(outputData, batchSize, 1)
}

// implements Fit function to train the model
func Fit(model *Linear, data *tensor.Tensor, labels *tensor.Tensor, epochs int, learningRate float64) {
	batchSize := data.Shape[0]
	numFeatures := len(model.Weights)

	for epoch := 0; epoch < epochs; epoch++ {

		totalLoss := 0.0

		for i := 0; i < batchSize; i++ {
			// get a single batch
			inputBatch := tensor.NewTensor(data.Data[i*numFeatures:(i+1)*numFeatures], 1, numFeatures)

			output := model.Forward(inputBatch)
			expected := tensor.NewTensor([]float64{labels.Data[i]}, 1)

			// Calculate loss
			lossTensor := nn.MSELoss(output, expected)
			loss := lossTensor.Data[0]
			totalLoss += loss

			// Gradient calculation for weights - replace this with SGD or something else
			for j := range model.Weights {
				model.Weights[j] -= learningRate * 2 * (output.Data[0] - expected.Data[0]) * inputBatch.Data[j]
			}

			// Gradient calculation for biases - replace this with SGD or something else
			for j := range model.Biases {
				model.Biases[j] -= learningRate * 2 * (output.Data[0] - expected.Data[0])
			}
		}

		avgLoss := totalLoss / float64(batchSize)
		fmt.Printf("Epoch [%d/%d], Loss: %.4f\n", epoch+1, epochs, avgLoss)
	}
}
