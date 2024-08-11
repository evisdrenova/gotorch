package nn

import (
	"gotorch/model"
)

type SGD struct {
	LearningRate float64
}

// implements stochastic gradient descent optimization function which updates the models weights and biases using the gradients computed during the backward pass
func (s *SGD) Step(model *model.Linear) {

	for i := range model.Weights {
		model.Weights[i] -= s.LearningRate * model.GradWeights[i]
	}

	for i := range model.Biases {
		model.Biases[i] -= s.LearningRate * model.GradBiases[i]
	}

}
