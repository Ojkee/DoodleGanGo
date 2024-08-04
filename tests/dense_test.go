package tests

import (
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
	"DoodleGan/layers"
)

func TestDenseLayer_1(t *testing.T) {
	layer := layers.NewDenseLayer(1, 2)
	weights := []float64{2, 3}
	layer.LoadWeights(&weights)
	input := *mat.NewVecDense(1, []float64{2})
	output := layer.Forward(input)
	target := *mat.NewVecDense(2, []float64{4, 6})
	if !functools.IsEqualVec(&target, output, 0.001) {
		t.Fatal()
	}
}

func TestDenseLayer_2(t *testing.T) {
	layer := layers.NewDenseLayer(3, 1)
	weights := []float64{-1, 2, 4}
	layer.LoadWeights(&weights)
	input := *mat.NewVecDense(3, []float64{-4, 8.5, 4})
	output := layer.Forward(input)
	target := *mat.NewVecDense(1, []float64{4 + 17 + 16})
	if !functools.IsEqualVec(&target, output, 0.001) {
		t.Fatal()
	}
}

func TestDenseLayer_3(t *testing.T) {
	layer := layers.NewDenseLayer(3, 1)
	weights := []float64{-1, 2, 4}
	layer.LoadWeights(&weights)
	layer.LoadBias(1)
	input := *mat.NewVecDense(3, []float64{-4, 8.5, 4})
	output := layer.Forward(input)
	target := *mat.NewVecDense(1, []float64{4 + 17 + 16 + 1})
	if !functools.IsEqualVec(&target, output, 0.001) {
		t.Fatal()
	}
}
