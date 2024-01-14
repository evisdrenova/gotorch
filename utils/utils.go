package utils

// checks if two slices are equal
// this is faster than doing reflect.deepCopy since we know the types and values
func AreSlicesEqual(s1, s2 []int) bool {

	if len(s1) != len(s2) {
		return false
	}

	for i, v := range s1 {
		if v != s2[i] {
			return false
		}
	}

	return true
}

func IsValidTensor(value interface{}) bool {

	return false

}
