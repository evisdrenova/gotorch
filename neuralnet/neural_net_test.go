package nn

import (
	"gotorch/model"
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

func TestSGDStep(t *testing.T) {
	initialWeights := []float64{0.5, -1.5}
	initialBiases := []float64{0.0}
	gradWeights := []float64{0.1, -0.2}
	gradBiases := []float64{0.05}

	model := &model.Linear{
		Weights:     initialWeights,
		Biases:      initialBiases,
		GradWeights: gradWeights,
		GradBiases:  gradBiases,
	}

	optimizer := &SGD{LearningRate: 0.1}

	expectedWeights := []float64{0.5 - 0.1*0.1, -1.5 - 0.1*(-0.2)} // {0.49, -1.48}
	expectedBiases := []float64{0.0 - 0.1*0.05}                    // {-0.005}

	optimizer.Step(model)

	// Check if the weights are updated correctly
	for i, weight := range model.Weights {
		if weight != expectedWeights[i] {
			t.Errorf("Weight %d: expected %f, got %f", i, expectedWeights[i], weight)
		}
	}

	// Check if the biases are updated correctly
	if model.Biases[0] != expectedBiases[0] {
		t.Errorf("Bias: expected %f, got %f", expectedBiases[0], model.Biases[0])
	}
}
