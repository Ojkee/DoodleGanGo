package optimizers_test

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
	"DoodleGan/optimizers"
)

func TestAdam_zero_momentum_zero_rho(t *testing.T) {
	conv1 := conv.NewConv2D([2]int{2, 1}, 2, [2]int{2, 2}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	act1 := conv.NewReLU()
	conv2 := conv.NewConv2D([2]int{1, 2}, 1, [2]int{1, 2}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})

	filter1 := []float64{
		1, -2, -1, 2,
		2, -1, 2, 1,
	}
	bias1 := []float64{
		1, -1,
	}
	filter2 := []float64{
		3, 1, -2, 2,
	}
	bias2 := []float64{
		2,
	}

	conv1.LoadFilter(&filter1)
	conv1.LoadBias(&bias1)
	conv2.LoadFilter(&filter2)
	conv2.LoadBias(&bias2)
	convs := []conv.ConvLayer{&conv1, &act1, &conv2}

	input1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2, 1, -2, 3}),
		*mat.NewDense(2, 2, []float64{1, -3, 4, 4}),
	}

	conv1.Forward(&input1)
	activated1 := act1.Forward(conv1.DeflatOutput())
	conv2.Forward(activated1)

	optimizer := optimizers.NewAdam(0.1, 0, 0, 1e-8)
	optimizer.PreTrainInit([2]int{1, 1}, &convs, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&convs, mat.NewVecDense(1, []float64{3}))

	resultFilter1_1 := conv1.GetFilter()
	targetFilter1_1 := []mat.Dense{
		*mat.NewDense(2, 1, []float64{0.9, -1.9}),
		*mat.NewDense(2, 1, []float64{-1, 1.9}),
		*mat.NewDense(2, 1, []float64{2.1, -1.1}),
		*mat.NewDense(2, 1, []float64{2.1, 1.1}),
	}
	resultFilter2_1 := conv2.GetFilter()
	targetFilter2_1 := []mat.Dense{
		*mat.NewDense(1, 2, []float64{2.9, 0.9}),
		*mat.NewDense(1, 2, []float64{-2.1, 2}),
	}
	resultBias1_1 := conv1.GetBias()
	targetBias1_1 := []float64{0.9, -0.9}
	resultBias2_1 := conv2.GetBias()
	targetBias2_1 := []float64{1.9}
	if !functools.IsEqualMatSlice(&targetFilter1_1, resultFilter1_1, 0.001) {
		fmt.Println("I FILTER")
		functools.PrintMatSlice(&targetFilter1_1, 3)
		functools.PrintMatSlice(resultFilter1_1, 3)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&targetFilter2_1, resultFilter2_1, 0.001) {
		fmt.Println("II FILTER")
		functools.PrintMatSlice(&targetFilter2_1, 3)
		functools.PrintMatSlice(resultFilter2_1, 3)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias1_1, resultBias1_1, 0.001) {
		fmt.Println("I BIAS")
		fmt.Println(&targetBias1_1)
		fmt.Println(resultBias1_1)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias2_1, resultBias2_1, 0.001) {
		fmt.Println("II BIAS")
		fmt.Println(&targetBias2_1)
		fmt.Println(resultBias2_1)
		t.Fail()
	}
}

func TestAdam_zero_momentum(t *testing.T) {
	conv1 := conv.NewConv2D([2]int{2, 1}, 2, [2]int{2, 2}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	act1 := conv.NewReLU()
	conv2 := conv.NewConv2D([2]int{1, 2}, 1, [2]int{1, 2}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})

	filter1 := []float64{
		1, -2, -1, 2,
		2, -1, 2, 1,
	}
	bias1 := []float64{
		1, -1,
	}
	filter2 := []float64{
		3, 1, -2, 2,
	}
	bias2 := []float64{
		2,
	}

	conv1.LoadFilter(&filter1)
	conv1.LoadBias(&bias1)
	conv2.LoadFilter(&filter2)
	conv2.LoadBias(&bias2)
	convs := []conv.ConvLayer{&conv1, &act1, &conv2}

	input1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2, 1, -2, 3}),
		*mat.NewDense(2, 2, []float64{1, -3, 4, 4}),
	}

	conv1.Forward(&input1)
	activated1 := act1.Forward(conv1.DeflatOutput())
	conv2.Forward(activated1)

	optimizer := optimizers.NewAdam(0.1, 0, 0.9, 1e-8)
	optimizer.PreTrainInit([2]int{1, 1}, &convs, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&convs, mat.NewVecDense(1, []float64{3}))

	fmt.Println("****************************")
	functools.PrintMatSlice(conv1.GetFilterGrads(), 2)
	fmt.Println(conv1.GetBiasGrads())
	fmt.Println("****************************")
	functools.PrintMatSlice(conv2.GetFilterGrads(), 2)
	fmt.Println(conv2.GetBiasGrads())
	fmt.Println("****************************")

	resultFilter1_1 := conv1.GetFilter()
	targetFilter1_1 := []mat.Dense{
		*mat.NewDense(2, 1, []float64{0.9, -1.9}),
		*mat.NewDense(2, 1, []float64{-1, 1.9}),
		*mat.NewDense(2, 1, []float64{2.1, -1.1}),
		*mat.NewDense(2, 1, []float64{2.1, 1.1}),
	}
	resultFilter2_1 := conv2.GetFilter()
	targetFilter2_1 := []mat.Dense{
		*mat.NewDense(1, 2, []float64{2.9, 0.9}),
		*mat.NewDense(1, 2, []float64{-2.1, 2}),
	}
	resultBias1_1 := conv1.GetBias()
	targetBias1_1 := []float64{0.9, -0.9}
	resultBias2_1 := conv2.GetBias()
	targetBias2_1 := []float64{1.9}
	if !functools.IsEqualMatSlice(&targetFilter1_1, resultFilter1_1, 0.001) {
		fmt.Println("I FILTER")
		functools.PrintMatSlice(&targetFilter1_1, 3)
		functools.PrintMatSlice(resultFilter1_1, 3)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&targetFilter2_1, resultFilter2_1, 0.001) {
		fmt.Println("II FILTER")
		functools.PrintMatSlice(&targetFilter2_1, 3)
		functools.PrintMatSlice(resultFilter2_1, 3)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias1_1, resultBias1_1, 0.001) {
		fmt.Println("I BIAS")
		fmt.Println(&targetBias1_1)
		fmt.Println(resultBias1_1)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias2_1, resultBias2_1, 0.001) {
		fmt.Println("II BIAS")
		fmt.Println(&targetBias2_1)
		fmt.Println(resultBias2_1)
		t.Fail()
	}
}
