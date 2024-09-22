package tests

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/optimizers"
)

func TestSDG_1(t *testing.T) {
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
	optimizer := optimizers.NewSDG(0.5, [2]int{outHeight, outWidth})
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
