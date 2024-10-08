package lf

import (
	"gotorch/tensor"
	"math"
	"testing"
)

func Test_MSELossError(t *testing.T) {

	input := tensor.NewTensor(4)
	target := tensor.NewTensor([]float64{9, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	_ = MSELoss(input, target)
}

func Test_MSELScalar(t *testing.T) {

	input := tensor.NewTensor(4)
	target := tensor.NewTensor(9)

	expectedMSE := (math.Pow(4-9, 2)) / 1

	result := MSELoss(input, target)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("MSELoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedMSE) > 1e-6 {
		t.Errorf("MSELoss incorrect, expected: %v, got: %v", expectedMSE, result.Data[0])
	}
}

func Test_MSELVector(t *testing.T) {

	input := tensor.NewTensor([]float64{2, -3, 4, -5})
	target := tensor.NewTensor([]float64{6, 23, -2, 8})

	expectedMSE := (math.Pow(2-6, 2) + math.Pow(-3-23, 2) + math.Pow(4-(-2), 2) + math.Pow(-5-8, 2)) / 4

	result := MSELoss(input, target)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("MSELoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedMSE) > 1e-6 {
		t.Errorf("MSELoss incorrect, expected: %v, got: %v", expectedMSE, result.Data[0])
	}
}

func Test_MSELMartix(t *testing.T) {

	input := tensor.NewTensor([][]float64{{2, -3}, {4, -5}})
	target := tensor.NewTensor([][]float64{{6, 23}, {-2, 8}})

	expectedMSE := (math.Pow(2-6, 2) + math.Pow(-3-23, 2) + math.Pow(4-(-2), 2) + math.Pow(-5-8, 2)) / 4

	result := MSELoss(input, target)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("MSELoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedMSE) > 1e-6 {
		t.Errorf("MSELoss incorrect, expected: %v, got: %v", expectedMSE, result.Data[0])
	}
}

func Test_BinaryCrossEntropyLossError(t *testing.T) {

	input := tensor.NewTensor(4)
	target := tensor.NewTensor([]float64{9, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	_ = BinaryCrossEntropyLoss(input, target)
}

func Test_BCELossVector(t *testing.T) {

	predictions := tensor.NewTensor([]float64{0.8, 0.1, 0.6, 0.3})
	target := tensor.NewTensor([]float64{1, 0, 1, 0})

	var expectedBCE float64
	for i := range predictions.Data {
		expectedBCE -= target.Data[i]*math.Log(predictions.Data[i]) + (1-target.Data[i])*math.Log(1-predictions.Data[i])
	}
	expectedBCE /= float64(len(predictions.Data))

	result := BinaryCrossEntropyLoss(predictions, target)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("BinaryCrossEntropyLoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedBCE) > 1e-6 {
		t.Errorf("BinaryCrossEntropyLoss incorrect, expected: %v, got: %v", expectedBCE, result.Data[0])
	}
}

func Test_BCELossMatrix(t *testing.T) {
	input := tensor.NewTensor([][]float64{{0.7, 0.2}, {0.6, 0.3}})
	target := tensor.NewTensor([][]float64{{1, 0}, {1, 0}})

	var expectedBCE float64
	for i := range input.Data {
		expectedBCE -= target.Data[i]*math.Log(input.Data[i]) + (1-target.Data[i])*math.Log(1-input.Data[i])
	}
	expectedBCE /= float64(len(input.Data))

	result := BinaryCrossEntropyLoss(input, target)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("BinaryCrossEntropyLoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedBCE) > 1e-6 {
		t.Errorf("BinaryCrossEntropyLoss incorrect, expected: %v, got: %v", expectedBCE, result.Data[0])
	}

}

func Test_CategoricalCrossEntropyLossError(t *testing.T) {

	input := tensor.NewTensor(4)
	target := tensor.NewTensor([]float64{9, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	_ = CategoricalCrossEntropyLoss(input, target)
}

func Test_CBCELossVector(t *testing.T) {
	predictions := tensor.NewTensor([]float64{0.8, 0.2})
	targets := tensor.NewTensor([]float64{1, 0})

	var expectedLoss float64
	var numClasses int

	if len(predictions.Shape) == 1 { // Vector case
		numClasses = len(predictions.Data)
		for c := 0; c < numClasses; c++ {
			expectedLoss -= targets.Data[c] * math.Log(predictions.Data[c])
		}
	}

	expectedLoss /= float64(len(predictions.Data) / numClasses)
	result := CategoricalCrossEntropyLoss(predictions, targets)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("CategoricalCrossEntropyLoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedLoss) > 1e-6 {
		t.Errorf("CategoricalCrossEntropyLoss incorrect, expected: %v, got: %v", expectedLoss, result.Data[0])
	}
}

func Test_CBCELossMatrix(t *testing.T) {

	predictions := tensor.NewTensor([][]float64{{0.7, 0.3}, {0.6, 0.4}})
	targets := tensor.NewTensor([][]float64{{1, 0}, {1, 0}})

	var expectedLoss float64
	numClasses := predictions.Shape[1]
	for i := 0; i < len(predictions.Data); i += numClasses {
		for c := 0; c < numClasses; c++ {
			expectedLoss -= targets.Data[i+c] * math.Log(predictions.Data[i+c])
		}
	}

	expectedLoss /= float64(len(predictions.Data) / numClasses)
	result := CategoricalCrossEntropyLoss(predictions, targets)

	if len(result.Shape) != 1 || result.Shape[0] != 1 {
		t.Errorf("CategoricalCrossEntropyLoss should return a scalar, got shape: %v", result.Shape)
	}

	if math.Abs(result.Data[0]-expectedLoss) > 1e-6 {
		t.Errorf("CategoricalCrossEntropyLoss incorrect, expected: %v, got: %v", expectedLoss, result.Data[0])
	}
}
