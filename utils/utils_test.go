package utils

import (
	"testing"

	"github.com/zeebo/assert"
)

func Test_areSlicesEqual_true(t *testing.T) {
	s1 := []int{1, 2, 3, 4}
	s2 := []int{1, 2, 3, 4}

	result := AreSlicesEqual(s1, s2)

	assert.True(t, result)

}

func Test_areSlicesEqual_false(t *testing.T) {
	s1 := []int{1, 2, 3, 4}
	s2 := []int{1, 2, 3}

	result := AreSlicesEqual(s1, s2)

	assert.False(t, result)

}
