package conv_test

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
)

// Single input
// Single filter
func TestConv2D_1(t *testing.T) {
	layer := conv.NewConv2D([2]int{3, 3}, 1, [2]int{4, 4}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{0, -1, 0, -1, 5, -1, 0, -1, 0}
	layer.LoadFilter(&filter)
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	layer.Forward(&[]mat.Dense{*mat.NewDense(4, 4, input)})
	if !reflect.DeepEqual(*layer.FlatOutput(), []float64{6, 7, 10, 11}) {
		t.Fatal()
	}
}

// Multi input
// Single filter
func TestConv2D_2(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 3}, 1, [2]int{2, 3}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		2, 2, 2, 2, 2, 2,
		1, 2, 3, 4, 5, 6,
	}
	layer.LoadFilter(&filter)
	layer.Forward(&[]mat.Dense{
		*mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}),
		*mat.NewDense(2, 3, []float64{3, 3, 3, 3, 3, 3}),
	})

	if !reflect.DeepEqual(*layer.FlatOutput(), []float64{105}) {
		t.Fatal()
	}
}

// Single input
// Multi filter
func TestConv2D_3(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 3, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 1,
		1, 1,
		-1, -1,
		-1, -1,
		2, 2,
		2, 2,
	}
	layer.LoadFilter(&filter)
	layer.Forward(&[]mat.Dense{*mat.NewDense(3, 3, []float64{1, 2, 1, 2, 3, 2, 1, 2, 1})})

	if !reflect.DeepEqual(
		*layer.FlatOutput(),
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
	layer := conv.NewConv2D([2]int{2, 2}, 3, [2]int{2, 2}, 4, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	layer.LoadFilter(&filter)

	input := []float64{
		1, 2, 3, 4,
		1, 3, 2, 4,
		1, -1, -1, 1,
		0, 5, 0, -1,
	}
	propperInput := layer.ArrayToConv2DInput(input)
	layer.Forward(&propperInput)
	target := []float64{4, 19, 4}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		t.Fatal()
	}
}

func TestConv2D_Stride_1(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{6, 6}, 1, [2]int{2, 2}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, -1,
		1, 1,
	}
	layer.LoadFilter(&filter)
	input := []float64{
		1, 0, 0, 1, 1, 1,
		1, 1, 1, 1, 0, 0,
		2, 5, 5, 2, 3, 4,
		1, 3, 1, 3, 1, 4,
		0, 1, 2, 1, 2, 0,
		2, 5, 4, 7, 0, 8,
	}
	propperInput := layer.ArrayToConv2DInput(input)
	layer.Forward(&propperInput)
	target := []float64{
		3, 1, 0,
		1, 7, 4,
		6, 12, 10,
	}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		t.Fatal()
	}
}

func TestConv2D_Stride_2(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{5, 5}, 1, [2]int{2, 2}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, -2,
		5, -3,
	}
	layer.LoadFilter(&filter)
	input := []float64{
		1, -2, 3, 1, 3,
		-3, 2, 3, 2, 1,
		-1, 0, 1, 2, 1,
		2, 1, 3, 4, -2,
		-2, 1, -3, 3, 1,
	}
	propperInput := layer.ArrayToConv2DInput(input)
	layer.Forward(&propperInput)
	target := []float64{
		-16, 10,
		6, 0,
	}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		fmt.Println(target)
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
}

func TestConv2D_Padding_1(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{1, 1, 0, 0})
	filter := []float64{
		2, -1,
		-3, 3,
	}
	layer.LoadFilter(&filter)
	input := []float64{
		3, 1, 5,
		5, -2, 1,
		-3, 4, -2,
	}
	propperInput := layer.ArrayToConv2DInput(input)
	layer.Forward(&propperInput)
	target := []float64{
		-6, 12, -15,
		-16, 6, 7,
		33, -23, 8,
	}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		fmt.Println(target)
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
}

func TestConv2D_Padding_2(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{2, 2, 1, 1})
	filter := []float64{
		1, -1,
		2, 1,
	}
	layer.LoadFilter(&filter)
	input := layer.ArrayToConv2DInput([]float64{
		1, 0, 3,
		0, 3, 1,
		3, 2, 1,
	})
	layer.Forward(&input)
	target := []float64{
		0, 0, 0, 0, 0,
		1, 2, 3, 6, 0,
		-1, 4, 4, 5, 0,
		3, 5, 7, 3, 0,
		-3, 1, 1, 1, 0,
	}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		fmt.Println(target)
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
}

func TestConv2D_All_1(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 3}, 2, [2]int{3, 3}, 3, [2]int{2, 1}, [4]int{1, 2, 0, 1})
	filter := []float64{
		1, 2, -1,
		3, 0, 1,

		2, 1, 0,
		0, 1, 2,

		-2, 0, -1,
		-2, 0, -1,

		2, 1, 0,
		2, 1, 0,

		-2, 1, 0,
		-2, 1, 0,

		1, 3, 1,
		-2, -2, -2,
	}
	layer.LoadFilter(&filter)
	input := layer.ArrayToConv2DInput([]float64{
		2, 1, -1,
		-2, 3, 1,
		3, 1, 2,

		-3, 2, 3,
		2, -2, 1,
		-2, 1, 2,

		4, 1, 2,
		-2, 1, 0,
		2, 1, 3,
	})
	layer.Forward(&input)
	target := []float64{
		1, 3, 4, -7,
		-6, 18, 3, 3,

		-11, -1, -6, -12,
		-10, -6, 9, -6,
	}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		fmt.Println(target)
		functools.PrintMatSlice(layer.DeflatOutput(), 0)
		t.Fatal()
	}
}

func TestConv2D_All_2(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 3}, 2, [2]int{3, 3}, 3, [2]int{2, 1}, [4]int{1, 2, 0, 1})
	filter := []float64{
		1, 2, -1,
		3, 0, 1,

		2, 1, 0,
		0, 1, 2,

		-2, 0, -1,
		-2, 0, -1,

		2, 1, 0,
		2, 1, 0,

		-2, 1, 0,
		-2, 1, 0,

		1, 3, 1,
		-2, -2, -2,
	}
	layer.LoadFilter(&filter)
	bias := []float64{
		1, -2,
	}
	layer.LoadBias(&bias)
	input := layer.ArrayToConv2DInput([]float64{
		2, 1, -1,
		-2, 3, 1,
		3, 1, 2,

		-3, 2, 3,
		2, -2, 1,
		-2, 1, 2,

		4, 1, 2,
		-2, 1, 0,
		2, 1, 3,
	})
	layer.Forward(&input)
	target := []float64{
		2, 4, 5, -6,
		-5, 19, 4, 4,

		-13, -3, -8, -14,
		-12, -8, 7, -8,
	}
	if !reflect.DeepEqual(target, *layer.FlatOutput()) {
		fmt.Println(target)
		functools.PrintMatSlice(layer.DeflatOutput(), 0)
		t.Fatal()
	}
}
