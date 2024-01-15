package data

import (
	"encoding/csv"
	"gotorch/tensor"
	"os"
	"reflect"
	"testing"
)

func Test_LoadFileFilepathError(t *testing.T) {

	filePath := "test32r23.csv"

	_, err := LoadCSV(filePath)

	if err == nil {
		t.Errorf("Should not be able to load the file")
	}
}

func Test_LoadUnableToReadFile(t *testing.T) {

	filePath := "test_error.csv"
	_, err := LoadCSV(filePath)

	if err == nil {
		t.Errorf("Should not be able to load the file")
	}
}

func Test_LoadUErrorParsing(t *testing.T) {

	filePath := "test_error_parsing.csv"
	_, err := LoadCSV(filePath)

	if err == nil {
		t.Errorf("Should not be able to load the file")
	}
}

func TestLoadCSV(t *testing.T) {
	expectedData := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	expectedShape := []int{3, 3}

	tensor, err := LoadCSV("test.csv")
	if err != nil {
		t.Fatalf("Failed to load CSV: %v", err)
	}

	if !reflect.DeepEqual(tensor.Data, expectedData) {
		t.Errorf("Incorrect data loaded. Expected %v, got %v", expectedData, tensor.Data)
	}

	if !reflect.DeepEqual(tensor.Shape, expectedShape) {
		t.Errorf("Incorrect shape of tensor. Expected %v, got %v", expectedShape, tensor.Shape)
	}
}

func Test_SaveCSV_Success(t *testing.T) {
	tensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	filePath := "success_test.csv"

	os.Remove(filePath)

	err := SaveCSV(tensor, filePath)
	if err != nil {
		t.Fatalf("Failed to save CSV: %v", err)
	}

	file, err := os.Open(filePath)
	if err != nil {
		t.Fatalf("Failed to open saved CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatalf("Failed to read saved CSV file: %v", err)
	}

	expected := [][]string{{"1", "2"}, {"3", "4"}}
	if !reflect.DeepEqual(records, expected) {
		t.Errorf("Incorrect CSV content. Expected %v, got %v", expected, records)
	}
}

func Test_SaveCSV_InvalidPath(t *testing.T) {
	tensor := tensor.NewTensor([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	invalidFilePath := "/invalidpath/test.csv"

	err := SaveCSV(tensor, invalidFilePath)
	if err == nil {
		t.Errorf("Expected an error due to invalid file path, but got none")
	}
}
