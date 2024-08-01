package tests

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/layers"
)

// Single input
// Single filter
func TestConv2D_1(t *testing.T) {
	layer := layers.NewConv2D([2]int{3, 3}, 1, [2]int{4, 4}, 1)
	filter := []float64{0, -1, 0, -1, 5, -1, 0, -1, 0}
	layer.LoadFilter(&filter)
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	layer.Forward([]mat.Dense{*mat.NewDense(4, 4, input)})
	if !reflect.DeepEqual(layer.FlatOutput(), []float64{6, 7, 10, 11}) {
		t.Fatal()
	}
}

// Multi input
// Single filter
func TestConv2D_2(t *testing.T) {
	layer := layers.NewConv2D([2]int{2, 3}, 1, [2]int{2, 3}, 2)
	filter := []float64{
		2, 2, 2, 2, 2, 2,
		1, 2, 3, 4, 5, 6,
	}
	layer.LoadFilter(&filter)
	layer.Forward([]mat.Dense{
		*mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}),
		*mat.NewDense(2, 3, []float64{3, 3, 3, 3, 3, 3}),
	})

	if !reflect.DeepEqual(layer.FlatOutput(), []float64{105}) {
		t.Fatal()
	}
}

// Single input
// Multi filter
func TestConv2D_3(t *testing.T) {
	layer := layers.NewConv2D([2]int{2, 2}, 3, [2]int{3, 3}, 1)
	filter := []float64{
		1, 1,
		1, 1,
		-1, -1,
		-1, -1,
		2, 2,
		2, 2,
	}
	layer.LoadFilter(&filter)
	layer.Forward([]mat.Dense{*mat.NewDense(3, 3, []float64{1, 2, 1, 2, 3, 2, 1, 2, 1})})

	if !reflect.DeepEqual(
		layer.FlatOutput(),
		[]float64{8, 8, 8, 8, -8, -8, -8, -8, 16, 16, 16, 16},
	) {
		t.Fatal()
	}
}

// Multi input
// Multi filter
func TestConv2D_4(t *testing.T) {
	filter := []float64{
		1, 1, 0, 0,
		1, 0, 0, 1,
		1, 0, 1, 0,
		1, -1, -1, -1,

		0, 1, 1, 0,
		0, 1, 0, 1,
		-1, -1, -1, 1,
		1, 1, 0, 0,

		1, 1, 0, 0,
		1, 0, 0, 1,
		1, 0, 1, 0,
		1, -1, -1, -1,
	}
	layer := layers.NewConv2D([2]int{2, 2}, 3, [2]int{2, 2}, 4)
	layer.LoadFilter(&filter)

	input := []float64{
		1, 2, 3, 4,
		1, 3, 2, 4,
		1, -1, -1, 1,
		0, 5, 0, -1,
	}
	propperInput := layer.ArrayToConv2DInput(input)
	layer.Forward(propperInput)
	target := []float64{4, 19, 4}
	if !reflect.DeepEqual(target, layer.FlatOutput()) {
		t.Fatal()
	}
}
