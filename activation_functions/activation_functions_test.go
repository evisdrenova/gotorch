package af

import (
	"gotorch/tensor"
	"math"
	"reflect"
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
func Test_SigmoidScalar(t *testing.T) {

	tensor := tensor.NewTensor(2)

	result := Sigmoid(tensor)

	expected := []float64{
		1 / (1 + math.Exp(-2)),
	}

	for i := range result.Data {
		if math.Abs(result.Data[i]-expected[i]) > 1e-6 { // using a small threshold for floating point comparison
			t.Errorf("Sigmoid function not applied correctly at index %d, expected: %v, got: %v", i, expected[i], result.Data[i])
		}
	}
}

func Test_SigmoidVector(t *testing.T) {

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

func Test_SigmoidMatrix(t *testing.T) {

	tensor := tensor.NewTensor([][]float64{{2, -3}, {4, -5}})

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

func Test_TanhScalar(t *testing.T) {

	tensor := tensor.NewTensor(2)

	result := Tanh(tensor)

	expected := []float64{
		(math.Exp(2-math.Exp(-2)) / (math.Exp(2) + math.Exp(-2))),
	}

	for i := range result.Data {
		if math.Abs(result.Data[i]-expected[i]) > 1e-6 { // using a small threshold for floating point comparison
			t.Errorf("Sigmoid function not applied correctly at index %d, expected: %v, got: %v", i, expected[i], result.Data[i])
		}
	}
}
func Test_TanhVector(t *testing.T) {

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

func Test_TanhMatrix(t *testing.T) {

	tensor := tensor.NewTensor([][]float64{{2, -3}, {4, -5}})

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

func Test_SoftmaxVector(t *testing.T) {

	tensor := tensor.NewTensor([]float64{2, 4, 5})

	total := math.Exp(2) + math.Exp(4) + math.Exp(5)
	expectedSM := []float64{
		math.Exp(2) / total,
		math.Exp(4) / total,
		math.Exp(5) / total,
	}

	result := SoftMax(tensor)

	if !reflect.DeepEqual(result.Shape, tensor.Shape) {
		t.Errorf("Softmax shape mismatch, expected: %v, got: %v", tensor.Shape, result.Shape)
	}
	for i, v := range result.Data {
		if math.Abs(v-expectedSM[i]) > 1e-6 {
			t.Errorf("Softmax value mismatch at index %d, expected: %v, got: %v", i, expectedSM[i], v)
		}
	}

}
