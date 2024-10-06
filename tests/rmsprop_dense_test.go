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

func TestRMSProp_Dense_1(t *testing.T) {
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

	input := mat.NewVecDense(3, []float64{-1, 0, 4})
	output_1 := dense_1.Forward(input)
	activated_1 := act_1.Forward(output_1)
	output_2 := dense_2.Forward(activated_1)
	act_2.Forward(output_2)

	nn := []layers.Layer{&dense_1, &act_1, &dense_2, &act_2}
	optimizers := optimizers.NewRMSProp(0.1, 0.0, 10e-8)
	vec_grad := mat.NewVecDense(3, []float64{0.5, 0.1, 0.2})
	optimizers.PreTrainInit([2]int{0, 0}, &[]conv.ConvLayer{}, &nn)
	optimizers.BackwardDenseLayers(&nn, vec_grad)

	fmt.Print("**************\n\n")
	functools.PrintMat(dense_1.GetOutWeightsGrads(), 3)
	fmt.Print("**************\n\n")
	functools.PrintMat(dense_2.GetOutWeightsGrads(), 3)
	fmt.Print("**************\n\n")

	result_weights_1 := dense_1.GetWeightsData()
	result_weights_2 := dense_2.GetWeightsData()
	target_weights_1 := []float64{0.91, 2, 0.36, -2.84, 2, 2.36}
	target_weights_2 := []float64{-2.05, 2.5, 0.99, 0.9, 1, -3}

	result_bias_1 := dense_1.GetBiasData()
	result_bias_2 := dense_2.GetBiasData()
	target_bias_1 := []float64{2.09, -5.16}
	target_bias_2 := []float64{-0.05, -0.01, 0}

	if !functools.IsEqual(&target_weights_1, &result_weights_1, 0.001) {
		fmt.Println("== I WEIGHTS ==")
		fmt.Println(target_weights_1)
		fmt.Println(result_weights_1)
		t.Fail()
	}
	if !functools.IsEqual(&target_weights_2, &result_weights_2, 0.001) {
		fmt.Println("== II WEIGHTS ==")
		fmt.Println(target_weights_2)
		fmt.Println(result_weights_2)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_1, &result_bias_1, 0.001) {
		fmt.Println("== I BIAS ==")
		fmt.Println(target_bias_1)
		fmt.Println(result_bias_1)
		t.Fail()
	}
	if !functools.IsEqual(&target_bias_2, &result_bias_2, 0.001) {
		fmt.Println("== II BIAS ==")
		fmt.Println(target_bias_2)
		fmt.Println(result_bias_2)
		t.Fail()
	}
}

// func TestRMSProp_Dense_Momentum_1(t *testing.T) {
// 	dense1 := layers.NewDenseLayer(3, 2)
// 	weights1 := []float64{3, 2, 1, -3, 2, 2}
// 	bias1 := []float64{2, -2}
// 	dense1.LoadWeights(&weights1)
// 	dense1.LoadBias(&bias1)
// 	act := layers.NewVReLU()
// 	dense2 := layers.NewDenseLayer(2, 1)
// 	weights2 := []float64{1, 2}
// 	bias2 := []float64{0}
// 	dense2.LoadWeights(&weights2)
// 	dense2.LoadBias(&bias2)
//
// 	denses := []layers.Layer{&dense1, &act, &dense2}
// 	optimizer := optimizers.NewRMSProp(0.1, 0.9, 10e-8)
// 	optimizer.PreTrainInit([2]int{0, 0}, &[]conv.ConvLayer{}, &denses)
//
// 	input1 := mat.NewVecDense(3, []float64{1, -2, 4})
// 	output1 := dense1.Forward(input1)
// 	activated := act.Forward(output1)
// 	dense2.Forward(activated)
//
// 	optimizer.BackwardDenseLayers(&denses, mat.NewVecDense(1, []float64{5}))
//
// 	resultWeights1 := dense1.GetWeights()
// 	targetWeights1 := mat.NewDense(2, 3, []float64{2.95, 2.1, 0.8, -3, 2, 2})
// 	resultBias1 := dense1.GetBias()
// 	targetBias1 := mat.NewVecDense(2, []float64{1.95, -2})
// 	resultWeights2 := dense2.GetWeights()
// 	targetWeights2 := mat.NewDense(1, 2, []float64{0.75, 2})
// 	resultBias2 := dense2.GetBias()
// 	targetBias2 := mat.NewVecDense(1, []float64{-0.05})
//
// 	if !reflect.DeepEqual(targetWeights1, resultWeights1) {
// 		fmt.Println("== I WEIGHTS == ")
// 		functools.PrintMat(targetWeights1, 3)
// 		functools.PrintMat(resultWeights1, 3)
// 		t.Fail()
// 	}
// 	if !reflect.DeepEqual(targetWeights2, resultWeights2) {
// 		fmt.Println("== II WEIGHTS == ")
// 		functools.PrintMat(targetWeights2, 3)
// 		functools.PrintMat(resultWeights2, 3)
// 		t.Fail()
// 	}
// 	if !functools.IsEqualVec(targetBias1, resultBias1, 0.001) {
// 		fmt.Println("== I BIAS ==")
// 		fmt.Println(targetBias1)
// 		fmt.Println(resultBias1)
// 		t.Fail()
// 	}
// 	if !functools.IsEqualVec(targetBias2, resultBias2, 0.001) {
// 		fmt.Println("== II BIAS ==")
// 		fmt.Println(targetBias2)
// 		fmt.Println(resultBias2)
// 		t.Fail()
// 	}
// }
//
// func TestRMSProp_Dense_Momentum_2(t *testing.T) {
// }
