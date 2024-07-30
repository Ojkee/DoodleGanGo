package tests

import (
	"fmt"
	"testing"

	"DoodleGan/layers"
)

func TestSoftmax(t *testing.T) {
	layer := layers.NewSoftmax()
	output := layer.Forward([]float64{
		1.1, 2.2, 0.2, -1.7,
	})
	target := []float64{
		0.224, 0.672, 0.091, 0.013,
	}
	fmt.Printf("output: %v\n", output)
	if !layers.IsEqual(&target, &output, 0.001) {
		t.Fatal()
	}
}
