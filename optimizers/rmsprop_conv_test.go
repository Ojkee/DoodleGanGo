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

func TestRMSProp_Conv_1(t *testing.T) {
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
	optimizer := optimizers.NewRMSProp(0.5, 0.0, 10e-8)
	convLayers := []conv.ConvLayer{&convLayer_1, &convLayer_2}
	optimizer.PreTrainInit([2]int{outHeight, outWidth}, &convLayers, &[]layers.Layer{})
	vec_grad := mat.NewVecDense(4, []float64{1, 0.5, -1, 1})
	optimizer.BackwardConv2DLayers(&convLayers, vec_grad)

	result_filter_1 := convLayer_1.GetFilter()
	result_bias_1 := convLayer_1.GetBias()
	result_filter_2 := convLayer_2.GetFilter()
	result_bias_2 := convLayer_2.GetBias()
	target_filter_1 := []mat.Dense{
		*mat.NewDense(3, 3, []float64{1.5, 1.5, -0.5, 0.5, 2.5, 1.5, 0.5, 0.5, 2.5}),
	}
	target_bias_1 := []float64{1}
	target_filter_2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1.5, -3.5, 0.5, 1.5}),
	}
	target_bias_2 := []float64{-0.5}

	if !functools.IsEqualMatSlice(&target_filter_1, result_filter_1, 0.001) {
		fmt.Println("== I FILTER ==")
		fmt.Println(&target_filter_1)
		fmt.Println(result_filter_1)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&target_filter_2, result_filter_2, 0.001) {
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

func TestRMSProp_Conv_2(t *testing.T) {
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
	optimizer := optimizers.NewRMSProp(0.1, 0.0, 10e-8)
	convLayers := []conv.ConvLayer{&conv_1, &act_1, &conv_2, &act_2, &conv_3}
	optimizer.PreTrainInit([2]int{outHeight, outWidth}, &convLayers, &[]layers.Layer{})

	vec_grad := mat.NewVecDense(1, []float64{-2})
	optimizer.BackwardConv2DLayers(&convLayers, vec_grad)

	target_filter_1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2.1, 0.9, 4.1, 0.9}),
	}
	target_filter_2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1.1, -1.9, 0.1, 1.1}),
	}
	target_filter_3 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1.1, 2, 0.9, 2.1}),
	}
	target_bias_1 := []float64{0.1}
	target_bias_2 := []float64{0.1}
	target_bias_3 := []float64{0.1}

	result_filter_1 := conv_1.GetFilter()
	result_filter_2 := conv_2.GetFilter()
	result_filter_3 := conv_3.GetFilter()
	result_bias_1 := conv_1.GetBias()
	result_bias_2 := conv_2.GetBias()
	result_bias_3 := conv_3.GetBias()

	if !functools.IsEqualMatSlice(&target_filter_1, result_filter_1, 0.001) {
		fmt.Println("== I FILTER ==")
		fmt.Println(&target_filter_1)
		fmt.Println(result_filter_1)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&target_filter_2, result_filter_2, 0.001) {
		fmt.Println("== II FILTER ==")
		fmt.Println(&target_filter_2)
		fmt.Println(result_filter_2)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&target_filter_3, result_filter_3, 0.001) {
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

func TestRMSProp_Conv_3(t *testing.T) {
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
	optimizer := optimizers.NewRMSProp(0.5, 0.0, 10e-8)
	optimizer.PreTrainInit([2]int{1, 1}, &cnn, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&cnn, grads)

	result_filter := conv_1.GetFilter()
	result_bias := conv_1.GetBias()
	target_filter := []mat.Dense{
		*mat.NewDense(2, 2, []float64{2.5, 2.5, -0.5, -1.5}),
		*mat.NewDense(2, 2, []float64{2.5, 1.5, 1., 1.5}),
		*mat.NewDense(2, 2, []float64{0.5, -1.5, -2.5, -0.5}),
		*mat.NewDense(2, 2, []float64{2.5, -1.5, -2, 2.5}),
	}
	target_bias := []float64{0.5, -1.5}
	if !functools.IsEqualMatSlice(&target_filter, result_filter, 0.001) {
		fmt.Println("== FILTER ==")
		functools.PrintMatSlice(&target_filter, 3)
		functools.PrintMatSlice(result_filter, 3)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias, result_bias, 0.001) {
		fmt.Println("== BIAS == ")
		fmt.Println(target_bias)
		fmt.Println(*result_bias)
		t.Fail()
	}
}

func TestRMSProp_Conv_4(t *testing.T) {
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
	optimizer := optimizers.NewRMSProp(0.1, 0.0, 10e-8)
	optimizer.PreTrainInit([2]int{1, 1}, &cnn, &[]layers.Layer{})
	optimizer.BackwardConv2DLayers(&cnn, grads)

	result_filter := conv_1.GetFilter()
	result_bias := conv_1.GetBias()
	target_filter := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.9, 1.9, -1.1, -2}),
		*mat.NewDense(2, 2, []float64{0, 1.9, 1.9, -3.9}),
	}
	target_bias := []float64{-0.1}
	if !functools.IsEqualMatSlice(&target_filter, result_filter, 0.001) {
		fmt.Println("== FILTER ==")
		functools.PrintMatSlice(&target_filter, 3)
		functools.PrintMatSlice(result_filter, 3)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias, result_bias, 0.001) {
		fmt.Println("== BIAS == ")
		fmt.Println(target_bias)
		fmt.Println(*result_bias)
		t.Fail()
	}
}

func TestRMSProp_Conv_Rho_1(t *testing.T) {
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
	optimizer := optimizers.NewRMSProp(0.1, 0.5, 10e-8)
	convLayers := []conv.ConvLayer{&convLayer_1, &act_1, &convLayer_2, &act_2}
	optimizer.PreTrainInit([2]int{outHeight, outWidth}, &convLayers, &[]layers.Layer{})
	vec_grad := mat.NewVecDense(2, []float64{1, 2})
	optimizer.BackwardConv2DLayers(&convLayers, vec_grad)

	result_filter_1 := convLayer_1.GetFilter()
	result_filter_2 := convLayer_2.GetFilter()
	target_filter_1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.8585, 2, 3.1414, 3.8585}),
		*mat.NewDense(2, 2, []float64{-4.1414, 2, -3.8585, 1.8585}),
		*mat.NewDense(2, 2, []float64{1.8585, -2, 2.1414, 1.8585}),
	}
	target_filter_2 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-2.1414, -0.1414, 1, 2.8585}),
		*mat.NewDense(2, 2, []float64{1.8585, -1.1414, 2, -0.1414}),
	}
	result_bias_1 := convLayer_1.GetBias()
	result_bias_2 := convLayer_2.GetBias()
	target_bias_1 := []float64{-0.1414}
	target_bias_2 := []float64{-0.1414, -0.1414}

	if !functools.IsEqualMatSlice(&target_filter_1, result_filter_1, 0.001) {
		fmt.Println("== I FILTER ==")
		functools.PrintMatSlice(&target_filter_1, 3)
		functools.PrintMatSlice(result_filter_1, 3)
		t.Fail()
	}
	if !functools.IsEqualMatSlice(&target_filter_2, result_filter_2, 0.001) {
		fmt.Println("== II FILTER ==")
		functools.PrintMatSlice(&target_filter_2, 3)
		functools.PrintMatSlice(result_filter_2, 3)
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

func TestRMSProp_Conv_Rho_2(t *testing.T) {
	convLayer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{2, 2}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{2, 0, 2, -1}
	convLayer.LoadFilter(&filter)
	bias := []float64{1}
	convLayer.LoadBias(&bias)
	convLayers := []conv.ConvLayer{&convLayer}

	optimizer := optimizers.NewRMSProp(0.5, 0.9, 10e-8)
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
	targetFilter := []mat.Dense{*mat.NewDense(2, 2, []float64{-1.0097, 3.0096, -0.3184, 1.7281})}
	resultBias := convLayer.GetBias()
	targetBias := []float64{-1.7282}

	if !functools.IsEqualMatSlice(&targetFilter, resultFilter, 0.001) {
		fmt.Println("== FILTER ==")
		functools.PrintMatSlice(&targetFilter, 4)
		functools.PrintMatSlice(resultFilter, 4)
		t.Fail()
	}
	if !functools.IsEqual(&targetBias, resultBias, 0.001) {
		fmt.Println("== BIAS ==")
		fmt.Println(&targetBias)
		fmt.Println(resultBias)
		t.Fail()
	}
}
