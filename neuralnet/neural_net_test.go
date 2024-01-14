package nn

import (
	"gotorch/tensor"
	"math"
	"testing"
)

func Test_ReLu(t *testing.T) {

	tensor := tensor.NewTensor([]float64{2, -3, 4, -5})

	result := ReLu(tensor)

	expected := []float64{2, 0, 4, 0}

	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("ReLu function not applied correctly, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}

}

func Test_Leaky_ReLu(t *testing.T) {

	tensor := tensor.NewTensor([]float64{2, -3, 4, -5})

	result := Leaky_ReLu(tensor)

	expected := []float64{2, -0.03, 4, -0.05}

	for i := range result.Data {
		if result.Data[i] != expected[i] {
			t.Errorf("ReLu function not applied correctly, expected: %v, got: %v", expected[i], result.Data[i])
		}
	}
}

func Test_Sigmoid(t *testing.T) {

	tensor := tensor.NewTensor([]float64{2, -3, 4, -5})

	result := Sigmoid(tensor)

	expected := []float64{
		1 / (1 + math.Exp(-2)),
		1 / (1 + math.Exp(3)),
		1 / (1 + math.Exp(-4)),
		1 / (1 + math.Exp(5)),
	}

	for i := range result.Data {
		if math.Abs(result.Data[i]-expected[i]) > 1e-6 { // using a small threshold for floating point comparison
			t.Errorf("Sigmoid function not applied correctly at index %d, expected: %v, got: %v", i, expected[i], result.Data[i])
		}
	}
}

func Test_Tanh(t *testing.T) {

	tensor := tensor.NewTensor([]float64{2, -3, 4, -5})

	result := Tanh(tensor)

	expected := []float64{
		(math.Exp(2-math.Exp(-2)) / (math.Exp(2) + math.Exp(-2))),
		(math.Exp(-3-math.Exp(3)) / (math.Exp(-3) + math.Exp(3))),
		(math.Exp(4-math.Exp(-4)) / (math.Exp(4) + math.Exp(-4))),
		(math.Exp(-5-math.Exp(5)) / (math.Exp(-5) + math.Exp(5))),
	}

	for i := range result.Data {
		if math.Abs(result.Data[i]-expected[i]) > 1e-6 { // using a small threshold for floating point comparison
			t.Errorf("Sigmoid function not applied correctly at index %d, expected: %v, got: %v", i, expected[i], result.Data[i])
		}
	}
}
