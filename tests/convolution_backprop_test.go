package tests

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
)

func TestConv2D_Backward_1(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2, -2, 0.5,
	}
	layer.LoadFilter(&filter)
	bias := []float64{1}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1, 2, 1, 0, 3, -2, 3, 4, 1}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{3, 2, 1, -2.5}),
	}

	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{3, 8, 4, -5, -3, -4, -2, 5.5, -1.25}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-0.5, 16, -1, 6.5}),
	}
	targetBiasGrads := []float64{3.5}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_2(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2, -2, 0.5,
		3, 1, 2, -3,
	}
	layer.LoadFilter(&filter)
	bias := []float64{1}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1, 2, 1, 0, 3, -2, 3, 4, 1}),
		*mat.NewDense(3, 3, []float64{2, 3, 2, 1, -2, 1, 2, 4, -1}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{3, 2, 1, -2.5}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{3, 8, 4, -5, -3, -4, -2, 5.5, -1.25}),
		*mat.NewDense(3, 3, []float64{9, 9, 2, 9, -11.5, -8.5, 2, -8, 7.5}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-0.5, 16, -1, 6.5}),
		*mat.NewDense(2, 2, []float64{18, 8.5, -9, 2.5}),
	}
	targetBiasGrads := []float64{3.5}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_3(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 2, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2, -2, 0.5,

		3, 1, 2, -3,
	}
	layer.LoadFilter(&filter)
	bias := []float64{1, 2}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1, 2, 1, 0, 3, -2, 3, 4, 1}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{3, 2, 1, -2.5}),
		*mat.NewDense(2, 2, []float64{2, 5, -1, 2}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{9, 25, 9, -4, 6, -17, -4, 12.5, -7.25}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-0.5, 16, -1, 6.5}),
		*mat.NewDense(2, 2, []float64{18, 2, 20, -6}),
	}
	targetBiasGrads := []float64{3.5, 8}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_4(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 3, [2]int{3, 3}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2, -2, 0.5,
		3, 1, 2, -3,

		4, -2, 3, -3,
		-1, 2, 2, -1,

		-1, 2, -1, -1,
		2, 0, -2, 0,
	}
	layer.LoadFilter(&filter)
	bias := []float64{1, 2, 3}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1, 2, 1, 0, 3, -2, 3, 4, 1}),
		*mat.NewDense(3, 3, []float64{3, 2, -2, 1, -1, 2, 5, -2, 0}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{3, 2, 1, -2.5}),
		*mat.NewDense(2, 2, []float64{0.5, 1, 2, 0.5}),
		*mat.NewDense(2, 2, []float64{2, 5, -1, 2}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()    // 2 Channels 3x3
	filterGrads := layer.GetFilterGrads() // 6 Channels 2x2
	biasGrads := layer.GetBiasGrads()     // 3 Channels 1x1 Flat

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{3, 10, 12, 3.5, -14.5, -9, 5, 0, -4.75}),
		*mat.NewDense(3, 3, []float64{12.5, 19, 4, 2, -12.5, -8.5, 8, -13, 7}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-0.5, 16, -1, 6.5}),
		*mat.NewDense(2, 2, []float64{16.5, -4, 11, -1}),
		*mat.NewDense(2, 2, []float64{4, 7, 11, 8}),
		*mat.NewDense(2, 2, []float64{5, -2, 8.5, -2.5}),
		*mat.NewDense(2, 2, []float64{18, 2, 20, -6}),
		*mat.NewDense(2, 2, []float64{13, -1, -12, 10}),
	}
	targetBiasGrads := []float64{3.5, 4, 8}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(*biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_5(t *testing.T) {
	layer := conv.NewConv2D([2]int{4, 4}, 1, [2]int{4, 4}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
	}
	layer.LoadFilter(&filter)
	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{1, 2, 1, 0, 3, -2, 3, 4, 1, 4, -1, 2, 5, 2, -2, 1}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(1, 1, []float64{0.1}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	var targetOutGradsTemp mat.Dense
	targetOutGradsTemp.Scale(
		0.1,
		mat.NewDense(4, 4, []float64{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}),
	)
	var targetFilterGradsTemp mat.Dense
	targetFilterGradsTemp.Scale(0.1, &input[0])
	targetOutGrads := []mat.Dense{targetOutGradsTemp}
	targetFilterGrads := []mat.Dense{targetFilterGradsTemp}
	targetBiasGrads := []float64{0.1}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(*biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_6(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{4, 4}, 1, [2]int{2, 2}, [4]int{0, 0, 0, 0})
	filter := []float64{2, -1, 2, 2}
	layer.LoadFilter(&filter)
	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{3, -1, 2, 1, -3, 1, 2, 2, 0.5, -0.5, 1, 2, 4, 2, 1, -4}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.5, 0, 1, -2}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(4, 4, []float64{1, -0.5, 0, 0, 1, 1, 0, 0, 2, -1, -4, 2, 2, 2, -4, -4}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0, -5, 0.5, 10.5}),
	}
	targetBiasGrads := []float64{-0.5}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(*biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_7(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{2, 2}, [4]int{0, 0, 0, 0})
	filter := []float64{
		2, 1, 1, 0,
	}
	layer.LoadFilter(&filter)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{3, -1, 8, -1, -2, 4, 5, -2, 1}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(1, 1, []float64{0.5}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1, 0.5, 0, 0.5, 0, 0, 0, 0, 0}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1.5, -0.5, -0.5, -1}),
	}
	targetBiasGrads := []float64{0.5}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(*biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}

func TestConv2D_Backward_8(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{2, 2}, [4]int{1, 1, 0, 0})
	filter := []float64{
		2, -1, 2, -1,
	}
	layer.LoadFilter(&filter)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{2, 1, -1, 3, 2, 4, -2, 1, 3}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2, -1, 1, 0.5}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{4, -2, -2, 2, -1, 1, 2, -1, 1}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2.5, -1, 2.5, 11}),
	}
	targetBiasGrads := []float64{2.5}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fail()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(*biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fail()
	}
}
