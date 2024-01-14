package nn

import (
	"gotorch/tensor"
	"testing"
)

func Test_ReLu(t *testing.T) {

	tensor := tensor.NewTensor([]float64{2, -3, 4, -5})

	result := ReLu(tensor)

	expected := []float64{2, 0, 4, 0}

	for i, _ := range result {
		if result[i] != expected[i] {
			t.Errorf("ReLu function not applied correctly, expected: %v, got: %v", expected[i], result[i])
		}
	}

}
