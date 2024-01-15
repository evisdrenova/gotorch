package data

import (
	"encoding/csv"
	"gotorch/tensor"
	"os"
	"strconv"
)

// load a csv file
// coudl add go funcs to this later to speed it up by chunking it
func LoadCSV(filepath string) (*tensor.Tensor, error) {

	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	// get size of file
	numRows := len(records)
	numCols := len(records[0])
	data := make([]float64, 0, numRows*numCols)

	// iterate over data and covert csv strings to float64s
	for _, record := range records {
		for _, field := range record {
			val, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, err
			}
			data = append(data, val)
		}
	}

	// create the new tensor
	tensor := tensor.NewTensor(data, numRows, numCols)

	return tensor, nil
}

// save a csv file
func SaveCSV(t *tensor.Tensor, filePath string) error {

	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// tensors are flattened by default, so need to reshape it
	numRows, numCols := t.Shape[0], t.Shape[1]
	dataIndex := 0

	for i := 0; i < numRows; i++ {
		record := make([]string, numCols)
		for j := 0; j < numCols; j++ {
			record[j] = strconv.FormatFloat(t.Data[dataIndex], 'f', -1, 64)
			dataIndex++
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil

}
