package nn

import (
	"gotorch/tensor"
	"reflect"
	"testing"
)

func Test_NewConv2D(t *testing.T) {
	inChannels := 3
	outChannels := 16
	kernelSize := 3
	stride := 1
	padding := 1

	conv := NewConv2D(inChannels, outChannels, kernelSize, stride, padding)

	// Check dimensions
	expectedFilterShape := []int{outChannels, inChannels, kernelSize, kernelSize}
	if !reflect.DeepEqual(conv.Filters.Shape, expectedFilterShape) {
		t.Errorf("Incorrect filter shape: got %v, want %v", conv.Filters.Shape, expectedFilterShape)
	}

	expectedBiasShape := []int{outChannels}
	if !reflect.DeepEqual(conv.Biases.Shape, expectedBiasShape) {
		t.Errorf("Incorrect bias shape: got %v, want %v", conv.Biases.Shape, expectedBiasShape)
	}
}

func Test_Conv2DForward(t *testing.T) {
	conv := NewConv2D(1, 1, 3, 1, 1)

	// Simple 4x4 input with batch size 1 and 1 channel
	input := tensor.NewTensor([]float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}, 1, 1, 4, 4)

	output := conv.Forward(input)

	// Check output dimensions
	expectedShape := []int{1, 1, 4, 4} // Same padding
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	// Verify output size matches expected dimensions
	expectedSize := 1 * 1 * 4 * 4 // batch * channels * height * width
	if len(output.Data) != expectedSize {
		t.Errorf("Incorrect output size: got %d, want %d", len(output.Data), expectedSize)
	}
}

func Test_Conv2DForwardMultiChannel(t *testing.T) {
	conv := NewConv2D(3, 2, 3, 1, 1)

	// 4x4 input with batch size 2 and 3 channels
	input := tensor.NewTensor(make([]float64, 2*3*4*4), 2, 3, 4, 4)

	output := conv.Forward(input)

	// Check output dimensions
	expectedShape := []int{2, 2, 4, 4} // batch, out_channels, height, width
	if !reflect.DeepEqual(output.Shape, expectedShape) {
		t.Errorf("Incorrect output shape: got %v, want %v", output.Shape, expectedShape)
	}

	expectedSize := 2 * 2 * 4 * 4 // batch * out_channels * height * width
	if len(output.Data) != expectedSize {
		t.Errorf("Incorrect output size: got %d, want %d", len(output.Data), expectedSize)
	}
}
