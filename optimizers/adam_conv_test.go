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

	optimizer := optimizers.NewAdam(0.1, 0.0, 0.9, 1e-8)
	optimizer.PreTrainInit([2]int{1, 1}, &convs, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&convs, mat.NewVecDense(1, []float64{3}))
	optimizer.UpdateCorrectionDecay()

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

	input2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 2, -5, 0}),
		*mat.NewDense(2, 2, []float64{3, -2, 2, 2}),
	}

	conv1.Forward(&input2)
	activated2 := act1.Forward(conv1.DeflatOutput())
	conv2.Forward(activated2)

	optimizer.BackwardConv2DLayers(&convs, mat.NewVecDense(1, []float64{2}))
	optimizer.UpdateCorrectionDecay()

	resultFilter1_2 := conv1.GetFilter()
	targetFilter1_2 := []mat.Dense{
		*mat.NewDense(2, 1, []float64{0.841, -1.768}),
		*mat.NewDense(2, 1, []float64{-1.138, 1.856}),
		*mat.NewDense(2, 1, []float64{2.056, -1.221}),
		*mat.NewDense(2, 1, []float64{2.233, 1.102}),
	}
	resultFilter2_2 := conv2.GetFilter()
	targetFilter2_2 := []mat.Dense{
		*mat.NewDense(1, 2, []float64{2.828, 0.811}),
		*mat.NewDense(1, 2, []float64{-2.196, 1.862}),
	}
	resultBias1_2 := conv1.GetBias()
	targetBias1_2 := []float64{0.823, -0.895}
	resultBias2_2 := conv2.GetBias()
	targetBias2_2 := []float64{1.820}
	if !functools.IsEqualMatSlice(&targetFilter1_2, resultFilter1_2, 0.001) {
		fmt.Println("I FILTER")
		functools.PrintMatSlice(&targetFilter1_2, 3)
		functools.PrintMatSlice(resultFilter1_2, 3)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&targetFilter2_2, resultFilter2_2, 0.001) {
		fmt.Println("II FILTER")
		functools.PrintMatSlice(&targetFilter2_2, 3)
		functools.PrintMatSlice(resultFilter2_2, 3)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias1_2, resultBias1_2, 0.001) {
		fmt.Println("I BIAS")
		fmt.Println(&targetBias1_2)
		fmt.Println(resultBias1_2)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias2_2, resultBias2_2, 0.001) {
		fmt.Println("II BIAS")
		fmt.Println(&targetBias2_2)
		fmt.Println(resultBias2_2)
		t.Fail()
	}
}

func TestAdam_zero_rho(t *testing.T) {
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

	optimizer := optimizers.NewAdam(0.1, 0.9, 0.0, 1e-8)
	optimizer.PreTrainInit([2]int{1, 1}, &convs, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&convs, mat.NewVecDense(1, []float64{3}))
	optimizer.UpdateCorrectionDecay()

	// fmt.Println("****************************")
	// functools.PrintMatSlice(conv1.GetFilterGrads(), 2)
	// fmt.Println(conv1.GetBiasGrads())
	// fmt.Println("****************************")
	// functools.PrintMatSlice(conv2.GetFilterGrads(), 2)
	// fmt.Println(conv2.GetBiasGrads())
	// fmt.Println("****************************")

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

	// input2 := []mat.Dense{
	// 	*mat.NewDense(2, 2, []float64{1, 2, -5, 0}),
	// 	*mat.NewDense(2, 2, []float64{3, -2, 2, 2}),
	// }
	//
	// conv1.Forward(&input2)
	// activated2 := act1.Forward(conv1.DeflatOutput())
	// conv2.Forward(activated2)
	//
	// optimizer.BackwardConv2DLayers(&convs, mat.NewVecDense(1, []float64{2}))
	// optimizer.UpdateCorrectionDecay()
	//
	// resultFilter1_2 := conv1.GetFilter()
	// targetFilter1_2 := []mat.Dense{
	// 	*mat.NewDense(2, 1, []float64{0.841, -1.768}),
	// 	*mat.NewDense(2, 1, []float64{-1.138, 1.856}),
	// 	*mat.NewDense(2, 1, []float64{2.056, -1.221}),
	// 	*mat.NewDense(2, 1, []float64{2.233, 1.102}),
	// }
	// resultFilter2_2 := conv2.GetFilter()
	// targetFilter2_2 := []mat.Dense{
	// 	*mat.NewDense(1, 2, []float64{2.828, 0.811}),
	// 	*mat.NewDense(1, 2, []float64{-2.196, 1.862}),
	// }
	// resultBias1_2 := conv1.GetBias()
	// targetBias1_2 := []float64{0.823, -0.895}
	// resultBias2_2 := conv2.GetBias()
	// targetBias2_2 := []float64{1.820}
	// if !functools.IsEqualMatSlice(&targetFilter1_2, resultFilter1_2, 0.001) {
	// 	fmt.Println("I FILTER")
	// 	functools.PrintMatSlice(&targetFilter1_2, 3)
	// 	functools.PrintMatSlice(resultFilter1_2, 3)
	// 	t.Fail()
	// }
	// if !functools.IsEqualMatSlice(&targetFilter2_2, resultFilter2_2, 0.001) {
	// 	fmt.Println("II FILTER")
	// 	functools.PrintMatSlice(&targetFilter2_2, 3)
	// 	functools.PrintMatSlice(resultFilter2_2, 3)
	// 	t.Fail()
	// }
	// if !functools.IsEqual(&targetBias1_2, resultBias1_2, 0.001) {
	// 	fmt.Println("I BIAS")
	// 	fmt.Println(&targetBias1_2)
	// 	fmt.Println(resultBias1_2)
	// 	t.Fail()
	// }
	// if !functools.IsEqual(&targetBias2_2, resultBias2_2, 0.001) {
	// 	fmt.Println("II BIAS")
	// 	fmt.Println(&targetBias2_2)
	// 	fmt.Println(resultBias2_2)
	// 	t.Fail()
	// }
}
