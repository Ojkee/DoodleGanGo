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
	conv_layer_1 := conv.NewConv2D(
		[2]int{3, 3},
		1,
		[2]int{5, 5},
		1,
		[2]int{1, 1},
		[4]int{0, 0, 0, 0},
	)
	filter_1 := []float64{2, 1, 0, 1, 2, 1, 0, 1, 2}
	conv_layer_1.LoadFilter(&filter_1)
	bias_1 := []float64{1}
	conv_layer_1.LoadBias(&bias_1)
	conv_layer_2 := conv.NewConv2D(
		[2]int{2, 2},
		1,
		[2]int{3, 3},
		1,
		[2]int{1, 1},
		[4]int{0, 0, 0, 0},
	)
	filter_2 := []float64{1, -3, 1, 1}
	conv_layer_2.LoadFilter(&filter_2)
	bias_2 := []float64{0}
	conv_layer_2.LoadBias(&bias_2)
	input := []mat.Dense{
		*mat.NewDense(5, 5, []float64{
			1, -2, 3, 0, 2,
			2, 1, -1, 3, -2,
			1, 3, -3, -1, 0,
			-2, 1, 2, 1, 3,
			3, 1, 0, 2, -1,
		}),
	}
	conv_layer_1.Forward(&input)
	conv_layer_2.Forward(conv_layer_1.DeflatOutput())

	outHeight, outWidth := conv_layer_2.GetOutputSize()
	optimizer := optimizers.NewSGD(0.5, [2]int{outHeight, outWidth})
	conv_layers := []conv.ConvLayer{&conv_layer_1, &conv_layer_2}
	vec_grad := mat.NewVecDense(4, []float64{1, 0.5, -1, 1})
	optimizer.BackwardConv2DLayers(&conv_layers, vec_grad)

	result_filter_1 := conv_layer_1.GetFilter()
	result_bias_1 := conv_layer_1.GetBias()
	result_filter_2 := conv_layer_2.GetFilter()
	result_bias_2 := conv_layer_2.GetBias()
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
	}
	if !functools.IsEqualMat(&target_filter_2, result_filter_2, 0.001) {
		fmt.Println("== II FILTER ==")
		fmt.Println(&target_filter_2)
		fmt.Println(result_filter_2)
	}
	if !functools.IsEqual(&target_bias_1, result_bias_1, 0.001) {
		fmt.Println("== I BIAS ==")
		fmt.Println(&target_bias_1)
		fmt.Println(result_bias_1)
	}
	if !functools.IsEqual(&target_bias_2, result_bias_2, 0.001) {
		fmt.Println("== II BIAS ==")
		fmt.Println(&target_bias_2)
		fmt.Println(result_bias_2)
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
	optimizer := optimizers.NewSGD(0.1, [2]int{outHeight, outWidth})
	conv_layers := []conv.ConvLayer{&conv_1, &act_1, &conv_2, &act_2, &conv_3}
	vec_grad := mat.NewVecDense(1, []float64{-2})
	optimizer.BackwardConv2DLayers(&conv_layers, vec_grad)

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
	}
	if !functools.IsEqualMat(&target_filter_2, result_filter_2, 0.001) {
		fmt.Println("== II FILTER ==")
		fmt.Println(&target_filter_2)
		fmt.Println(result_filter_2)
	}
	if !functools.IsEqualMat(&target_filter_3, result_filter_3, 0.001) {
		fmt.Println("== III FILTER ==")
		fmt.Println(&target_filter_3)
		fmt.Println(result_filter_3)
	}
	if !functools.IsEqual(&target_bias_1, result_bias_1, 0.001) {
		fmt.Println("== I BIAS ==")
		fmt.Println(&target_bias_1)
		fmt.Println(result_bias_1)
	}
	if !functools.IsEqual(&target_bias_2, result_bias_2, 0.001) {
		fmt.Println("== II BIAS ==")
		fmt.Println(&target_bias_2)
		fmt.Println(result_bias_2)
	}
	if !functools.IsEqual(&target_bias_3, result_bias_3, 0.001) {
		fmt.Println("== III BIAS ==")
		fmt.Println(&target_bias_3)
		fmt.Println(result_bias_3)
	}
}

func TestSGD_Dense_1(t *testing.T) {
	dense_1 := layers.NewDenseLayer(3, 2)
	filter_1 := []float64{1, 2, 0, -3, 2, 3}
	bias_1 := []float64{2, -5}
	dense_1.LoadWeights(&filter_1)
	dense_1.LoadBias(&bias_1)
	act_1 := layers.NewVReLU()

	dense_2 := layers.NewDenseLayer(2, 3)
	filter_2 := []float64{-2, 3, 1, 1, 1, -3}
	bias_2 := []float64{0, 0, 0}
	dense_2.LoadWeights(&filter_2)
	dense_2.LoadBias(&bias_2)
	act_2 := layers.NewVReLU()

	// input := mat.NewVecDense(3, []float64{-1, 0, 4})
	nn := []layers.Layer{&dense_1, &act_1, &dense_2, &act_2}
	fmt.Printf("%v\n", nn)
}
