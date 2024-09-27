package tests

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
	"DoodleGan/optimizers"
)

func TestSGD_Conv_1(t *testing.T) {
	convLayer_1 := conv.NewConv2D(
		[2]int{3, 3},
		1,
		[2]int{5, 5},
		1,
		[2]int{1, 1},
		[4]int{0, 0, 0, 0},
	)
	filter_1 := []float64{2, 1, 0, 1, 2, 1, 0, 1, 2}
	convLayer_1.LoadFilter(&filter_1)
	bias_1 := []float64{1}
	convLayer_1.LoadBias(&bias_1)
	convLayer_2 := conv.NewConv2D(
		[2]int{2, 2},
		1,
		[2]int{3, 3},
		1,
		[2]int{1, 1},
		[4]int{0, 0, 0, 0},
	)
	filter_2 := []float64{1, -3, 1, 1}
	convLayer_2.LoadFilter(&filter_2)
	bias_2 := []float64{0}
	convLayer_2.LoadBias(&bias_2)
	input := []mat.Dense{
		*mat.NewDense(5, 5, []float64{
			1, -2, 3, 0, 2,
			2, 1, -1, 3, -2,
			1, 3, -3, -1, 0,
			-2, 1, 2, 1, 3,
			3, 1, 0, 2, -1,
		}),
	}
	convLayer_1.Forward(&input)
	convLayer_2.Forward(convLayer_1.DeflatOutput())

	outHeight, outWidth := convLayer_2.GetOutputSize()
	optimizer := optimizers.NewSGD(0.5, 0.0)
	convLayers := []conv.ConvLayer{&convLayer_1, &convLayer_2}
	optimizer.PreTrainInit([2]int{outHeight, outWidth}, &convLayers, &[]layers.Layer{})
	vec_grad := mat.NewVecDense(4, []float64{1, 0.5, -1, 1})
	optimizer.BackwardConv2DLayers(&convLayers, vec_grad)

	result_filter_1 := convLayer_1.GetFilter()
	result_bias_1 := convLayer_1.GetBias()
	result_filter_2 := convLayer_2.GetFilter()
	result_bias_2 := convLayer_2.GetBias()
	target_filter_1 := []mat.Dense{
		*mat.NewDense(3, 3, []float64{-0.75, 14.25, -12.25, -13.5, 9.5, 6, 2.25, -9.75, 3.75}),
	}
	target_bias_1 := []float64{1}
	target_filter_2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{7.75, -4.75, -9.5, 5.5}),
	}
	target_bias_2 := []float64{-0.75}

	if !functools.IsEqualMat(&target_filter_1, result_filter_1, 0.001) {
		fmt.Println("== I FILTER ==")
		fmt.Println(&target_filter_1)
		fmt.Println(result_filter_1)
		t.Fail()
	}
	if !functools.IsEqualMat(&target_filter_2, result_filter_2, 0.001) {
		fmt.Println("== II FILTER ==")
		fmt.Println(&target_filter_2)
		fmt.Println(result_filter_2)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_1, result_bias_1, 0.001) {
		fmt.Println("== I BIAS ==")
		fmt.Println(&target_bias_1)
		fmt.Println(result_bias_1)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_2, result_bias_2, 0.001) {
		fmt.Println("== II BIAS ==")
		fmt.Println(&target_bias_2)
		fmt.Println(result_bias_2)
		t.Fail()
	}
}

func TestSGD_Conv_2(t *testing.T) {
	conv_1 := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter_1 := []float64{2, 1, 4, 1}
	conv_1.LoadFilter(&filter_1)
	act_1 := conv.NewReLU()
	conv_2 := conv.NewConv2D([2]int{2, 2}, 1, [2]int{2, 2}, 1, [2]int{1, 1}, [4]int{1, 1, 0, 0})
	filter_2 := []float64{1, -2, 0, 1}
	conv_2.LoadFilter(&filter_2)
	act_2 := conv.NewLeakyReLU(0.1)
	conv_3 := conv.NewConv2D([2]int{2, 2}, 1, [2]int{2, 2}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter_3 := []float64{1, 2, 1, 2}
	conv_3.LoadFilter(&filter_3)

	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{-2, 1, 0, 0, 2, -1, 0, -1, 2}),
	}
	conv_1.Forward(&input)
	activated_1 := act_1.Forward(conv_1.DeflatOutput())
	conv_2.Forward(activated_1)
	activated_2 := act_2.Forward(conv_2.DeflatOutput())
	conv_3.Forward(activated_2)

	outHeight, outWidth := conv_3.GetOutputSize()
	optimizer := optimizers.NewSGD(0.1, 0.0)
	convLayers := []conv.ConvLayer{&conv_1, &act_1, &conv_2, &act_2, &conv_3}
	optimizer.PreTrainInit([2]int{outHeight, outWidth}, &convLayers, &[]layers.Layer{})

	vec_grad := mat.NewVecDense(1, []float64{-2})
	optimizer.BackwardConv2DLayers(&convLayers, vec_grad)

	target_filter_1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2.6, 0.98, 5.1, 0.48}),
	}
	target_filter_2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{4.6, -1.82, 0.78, 2.82}),
	}
	target_filter_3 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2.8, 2, 0.66, 3.8}),
	}
	target_bias_1 := []float64{0.58}
	target_bias_2 := []float64{0.66}
	target_bias_3 := []float64{0.2}

	result_filter_1 := conv_1.GetFilter()
	result_filter_2 := conv_2.GetFilter()
	result_filter_3 := conv_3.GetFilter()
	result_bias_1 := conv_1.GetBias()
	result_bias_2 := conv_2.GetBias()
	result_bias_3 := conv_3.GetBias()

	if !functools.IsEqualMat(&target_filter_1, result_filter_1, 0.001) {
		fmt.Println("== I FILTER ==")
		fmt.Println(&target_filter_1)
		fmt.Println(result_filter_1)
		t.Fail()
	}
	if !functools.IsEqualMat(&target_filter_2, result_filter_2, 0.001) {
		fmt.Println("== II FILTER ==")
		fmt.Println(&target_filter_2)
		fmt.Println(result_filter_2)
		t.Fail()
	}
	if !functools.IsEqualMat(&target_filter_3, result_filter_3, 0.001) {
		fmt.Println("== III FILTER ==")
		fmt.Println(&target_filter_3)
		fmt.Println(result_filter_3)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_1, result_bias_1, 0.001) {
		fmt.Println("== I BIAS ==")
		fmt.Println(&target_bias_1)
		fmt.Println(result_bias_1)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_2, result_bias_2, 0.001) {
		fmt.Println("== II BIAS ==")
		fmt.Println(&target_bias_2)
		fmt.Println(result_bias_2)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_3, result_bias_3, 0.001) {
		fmt.Println("== III BIAS ==")
		fmt.Println(&target_bias_3)
		fmt.Println(result_bias_3)
		t.Fail()
	}
}

func TestSGD_Conv_3(t *testing.T) {
	conv_1 := conv.NewConv2D([2]int{2, 2}, 2, [2]int{4, 4}, 2, [2]int{2, 2}, [4]int{0, 0, 0, 0})
	filter := []float64{
		3, 3, 0, -1, 2, 2, 1, 2,
		1, -1, -2, 0, 2, -1, -2, 3,
	}
	conv_1.LoadFilter(&filter)
	bias_1 := []float64{1, -1}
	conv_1.LoadBias(&bias_1)
	pool_1 := conv.NewAvgPool([2]int{2, 2}, [2]int{2, 2}, [2]int{2, 2})
	act_1 := conv.NewLeakyReLU(0.1)

	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			-1, 3, 0, 4, -1, 1, 3, 1, 4, 0, 4, -3, -2, -1, 2, 3,
		}),
		*mat.NewDense(4, 4, []float64{
			-3, 0, -3, -3, 2, 1, 0, 0, -1, 1, 1, 3, -1, -1, -1, 3,
		}),
	}
	conv_1.Forward(&input)
	pooled := pool_1.Forward(conv_1.DeflatOutput())
	act_1.Forward(pooled)

	cnn := []conv.ConvLayer{&conv_1, &pool_1, &act_1}
	grads := mat.NewVecDense(2, []float64{1, 2})
	optimizer := optimizers.NewSGD(0.5, 0.0)
	optimizer.PreTrainInit([2]int{1, 1}, &cnn, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&cnn, grads)

	result_filter := conv_1.GetFilter()
	result_bias := conv_1.GetBias()
	target_filter := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2.125, 2.5, -0.25, -1.5}),
		*mat.NewDense(2, 2, []float64{2.75, 1.875, 1, 1.625}),
		*mat.NewDense(2, 2, []float64{0.825, -1.1, -2.05, -0.1}),
		*mat.NewDense(2, 2, []float64{2.15, -1.025, -2, 2.925}),
	}
	target_bias := []float64{0.5, -1.1}
	if !functools.IsEqualMat(&target_filter, result_filter, 0.001) {
		fmt.Println("== FILTER ==")
		functools.PrintMatArray(&target_filter, 3)
		functools.PrintMatArray(result_filter, 3)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias, result_bias, 0.001) {
		fmt.Println("== BIAS == ")
		fmt.Println(target_bias)
		fmt.Println(*result_bias)
		t.Fail()
	}
}

func TestSGD_Conv_4(t *testing.T) {
	conv_1 := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2, -1, -2,
		0, 2, 2, -4,
	}
	conv_1.LoadFilter(&filter)
	pool := conv.NewMaxPool([2]int{2, 2}, [2]int{2, 2}, [2]int{2, 2}, 1)

	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1, 3, -1, 2, 3, -2, 1, 0, 4}),
		*mat.NewDense(3, 3, []float64{1, 2, 4, 0, 3, 3, 2, -2, -2}),
	}
	conv_1.Forward(&input)
	pool.Forward(conv_1.DeflatOutput())

	cnn := []conv.ConvLayer{&conv_1, &pool}
	grads := mat.NewVecDense(1, []float64{5})
	optimizer := optimizers.NewSGD(0.1, 0.0)
	optimizer.PreTrainInit([2]int{1, 1}, &cnn, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&cnn, grads)

	result_filter := conv_1.GetFilter()
	result_bias := conv_1.GetBias()
	target_filter := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0, 0.5, -1.5, -2}),
		*mat.NewDense(2, 2, []float64{0, 0.5, 1, -3}),
	}
	target_bias := []float64{-0.5}
	if !functools.IsEqualMat(&target_filter, result_filter, 0.001) {
		fmt.Println("== FILTER ==")
		functools.PrintMatArray(&target_filter, 3)
		functools.PrintMatArray(result_filter, 3)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias, result_bias, 0.001) {
		fmt.Println("== BIAS == ")
		fmt.Println(target_bias)
		fmt.Println(*result_bias)
		t.Fail()
	}
}

func TestSGD_Conv_Momentum_1(t *testing.T) {
	convLayer_1 := conv.NewConv2D(
		[2]int{2, 2},
		1,
		[2]int{1, 1},
		3,
		[2]int{1, 1},
		[4]int{1, 1, 1, 1},
	)
	filter_1 := []float64{1, 2, 3, 4, -4, 2, -4, 2, 2, -2, 2, 2}
	convLayer_1.LoadFilter(&filter_1)
	act_1 := conv.NewReLU()
	convLayer_2 := conv.NewConv2D(
		[2]int{2, 2},
		2,
		[2]int{2, 2},
		1,
		[2]int{1, 1},
		[4]int{0, 0, 0, 0},
	)
	filter_2 := []float64{-2, 0, 1, 3, 2, -1, 2, 0}
	convLayer_2.LoadFilter(&filter_2)
	act_2 := conv.NewLeakyReLU(0.1)

	input := []mat.Dense{
		*mat.NewDense(1, 1, []float64{2}),
		*mat.NewDense(1, 1, []float64{0.5}),
		*mat.NewDense(1, 1, []float64{3}),
	}
	convLayer_1.Forward(&input)
	activated_1 := act_1.Forward(convLayer_1.DeflatOutput())
	convLayer_2.Forward(activated_1)
	act_2.Forward(convLayer_2.DeflatOutput())

	outHeight, outWidth := convLayer_2.GetOutputSize()
	optimizer := optimizers.NewSGD(0.1, 0.5)
	convLayers := []conv.ConvLayer{&convLayer_1, &act_1, &convLayer_2, &act_2}
	optimizer.PreTrainInit([2]int{outHeight, outWidth}, &convLayers, &[]layers.Layer{})
	vec_grad := mat.NewVecDense(2, []float64{1, 2})
	optimizer.BackwardConv2DLayers(&convLayers, vec_grad)

	result_filter_1 := convLayer_1.GetFilter()
	result_filter_2 := convLayer_2.GetFilter()
	target_filter_1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.97, 2, 3.2, 3.62}),
		*mat.NewDense(2, 2, []float64{-4.0075, 2, -3.95, 1.905}),
		*mat.NewDense(2, 2, []float64{1.955, -2, 2.3, 1.43}),
	}
	target_filter_2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-2.075, -0.05, 1, 2.97}),
		*mat.NewDense(2, 2, []float64{0.5, -2, 2, -0.6}),
	}
	result_bias_1 := convLayer_1.GetBias()
	result_bias_2 := convLayer_2.GetBias()
	target_bias_1 := []float64{-0.105}
	target_bias_2 := []float64{-0.005, -0.1}

	if !functools.IsEqualMat(&target_filter_1, result_filter_1, 0.0001) {
		fmt.Println("== I FILTER ==")
		fmt.Println(&target_filter_1)
		fmt.Println(result_filter_1)
		t.Fail()
	}
	if !functools.IsEqualMat(&target_filter_2, result_filter_2, 0.0001) {
		fmt.Println("== II FILTER ==")
		fmt.Println(&target_filter_2)
		fmt.Println(result_filter_2)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_1, result_bias_1, 0.0001) {
		fmt.Println("== I BIAS ==")
		fmt.Println(&target_bias_1)
		fmt.Println(result_bias_1)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_2, result_bias_2, 0.0001) {
		fmt.Println("== II BIAS ==")
		fmt.Println(&target_bias_2)
		fmt.Println(result_bias_2)
		t.Fail()
	}
}

func TestSGD_Conv_Momentum_2(t *testing.T) {
	convLayer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{2, 2}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{2, 0, 2, -1}
	convLayer.LoadFilter(&filter)
	bias := []float64{1}
	convLayer.LoadBias(&bias)
	convLayers := []conv.ConvLayer{&convLayer}

	optimizer := optimizers.NewSGD(0.5, 0.9)
	optimizer.PreTrainInit([2]int{1, 1}, &convLayers, &[]layers.Layer{})

	input1 := []mat.Dense{*mat.NewDense(2, 2, []float64{1, -1, 2, -3})}
	convLayer.Forward(&input1)
	grad1 := mat.NewVecDense(1, []float64{2})
	optimizer.BackwardConv2DLayers(&convLayers, grad1)

	input2 := []mat.Dense{*mat.NewDense(2, 2, []float64{2, -2, 1, -3})}
	convLayer.Forward(&input2)
	grad2 := mat.NewVecDense(1, []float64{2})
	optimizer.BackwardConv2DLayers(&convLayers, grad2)

	resultFilter := convLayer.GetFilter()
	targetFilter := []mat.Dense{*mat.NewDense(2, 2, []float64{1.61, 0.39, 1.52, -0.13})}
	resultBias := convLayer.GetBias()
	targetBias := []float64{0.71}

	if !functools.IsEqualMat(&targetFilter, resultFilter, 0.001) {
		fmt.Println("== FILTER ==")
		functools.PrintMatArray(&targetFilter, 4)
		functools.PrintMatArray(resultFilter, 4)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias, resultBias, 0.001) {
		fmt.Println("== BIAS ==")
		fmt.Println(&targetBias)
		fmt.Println(resultBias)
		t.Fail()
	}
}
