package tests

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
	"DoodleGan/layers"
)

func TestDenseLayer_Forward_1(t *testing.T) {
	layer := layers.NewDenseLayer(1, 2)
	weights := []float64{2, 3}
	layer.LoadWeights(&weights)
	input := *mat.NewVecDense(1, []float64{2})
	output := layer.Forward(&input)
	target := *mat.NewVecDense(2, []float64{4, 6})
	if !functools.IsEqualVec(&target, output, 0.001) {
		t.Fatal()
	}
}

func TestDenseLayer_Forward_2(t *testing.T) {
	layer := layers.NewDenseLayer(3, 1)
	weights := []float64{-1, 2, 4}
	layer.LoadWeights(&weights)
	input := *mat.NewVecDense(3, []float64{-4, 8.5, 4})
	output := layer.Forward(&input)
	target := *mat.NewVecDense(1, []float64{4 + 17 + 16})
	if !functools.IsEqualVec(&target, output, 0.001) {
		t.Fatal()
	}
}

func TestDenseLayer_Forward_3(t *testing.T) {
	layer := layers.NewDenseLayer(3, 1)
	weights := []float64{-1, 2, 4}
	layer.LoadWeights(&weights)
	bias := []float64{1}
	layer.LoadBias(&bias)
	input := *mat.NewVecDense(3, []float64{-4, 8.5, 4})
	output := layer.Forward(&input)
	target := *mat.NewVecDense(1, []float64{4 + 17 + 16 + 1})
	if !functools.IsEqualVec(&target, output, 0.001) {
		t.Fatal()
	}
}

func TestDenseLayer_Forward_4(t *testing.T) {
	layer := layers.NewDenseLayer(3, 4)
	weights := []float64{
		-1, 2, 3,
		0.5, 2, -1,
		2, -2, 1,
		3, 3, -3,
	}
	layer.LoadWeights(&weights)
	bias := []float64{
		0.5, 1, 0, 2,
	}
	layer.LoadBias(&bias)
	input := mat.NewVecDense(3, []float64{
		1, 2, 3,
	})
	output := layer.Forward(input)
	target := []float64{
		12.5, 2.5, 1, 2,
	}
	if !reflect.DeepEqual(target, output.RawVector().Data) {
		fmt.Println(target)
		fmt.Println(output.RawVector().Data)
		t.Fatal()
	}
}

func TestDenseLayer_Backward_1(t *testing.T) {
	layer := layers.NewDenseLayer(2, 1)
	weights := []float64{
		2,
		1,
	}
	layer.LoadWeights(&weights)
	bias := []float64{
		0.5,
	}
	layer.LoadBias(&bias)
	input := mat.NewVecDense(2, []float64{
		0.5, -0.5,
	})
	layer.Forward(input)
	inGrads := []float64{
		4,
	}
	outGrads := layer.Backward(mat.NewVecDense(1, inGrads))
	targetWeightsGrads := mat.NewDense(1, 2, []float64{
		2, -2,
	})
	targetBiasGrads := mat.NewVecDense(1, []float64{
		4,
	})
	targetOutGrads := mat.NewVecDense(2, []float64{
		8, 4,
	})

	if !reflect.DeepEqual(targetOutGrads, outGrads) {
		fmt.Println(targetOutGrads)
		fmt.Println(outGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(targetWeightsGrads, layer.GetOutWeightsGrads()) {
		fmt.Println(targetWeightsGrads)
		fmt.Println(layer.GetOutWeightsGrads())
		t.Fatal()
	}
	if !reflect.DeepEqual(targetBiasGrads, layer.GetOutBiasGrads()) {
		fmt.Println(targetBiasGrads)
		fmt.Println(layer.GetOutBiasGrads())
		t.Fatal()
	}
}

func TestDenseLayer_Backward_2(t *testing.T) {
	layer := layers.NewDenseLayer(3, 2)
	weights := []float64{
		2, 1, -2,
		-0.5, 0.75, 3,
	}
	layer.LoadWeights(&weights)
	bias := []float64{
		0.5, -0.5,
	}
	layer.LoadBias(&bias)
	input := mat.NewVecDense(3, []float64{
		0.5, -0.5, 1,
	})
	layer.Forward(input)
	outGrads := layer.Backward(mat.NewVecDense(2, []float64{
		4, -2,
	}))
	targetWeightsGrads := mat.NewDense(2, 3, []float64{
		2, -2, 4,
		-1, 1, -2,
	})
	targetBiasGrads := mat.NewVecDense(2, []float64{
		4, -2,
	})
	targetOutGrads := mat.NewVecDense(3, []float64{
		9, 2.5, -14,
	})
	if !reflect.DeepEqual(targetOutGrads, outGrads) {
		fmt.Println("--OUT GRADS--")
		fmt.Println(targetOutGrads)
		fmt.Println(outGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(targetWeightsGrads, layer.GetOutWeightsGrads()) {
		fmt.Println("--WEIGHT GRADS--")
		fmt.Println(targetWeightsGrads)
		fmt.Println(layer.GetOutWeightsGrads())
		t.Fatal()
	}
	if !reflect.DeepEqual(targetBiasGrads, layer.GetOutBiasGrads()) {
		fmt.Println("--BIAS GRADS--")
		fmt.Println(targetBiasGrads)
		fmt.Println(layer.GetOutBiasGrads())
		t.Fatal()
	}

	learningRate := 0.5
	layer.ApplyGrads(&learningRate, layer.GetOutWeightsGrads(), layer.GetOutBiasGrads())
	targetUpdatedWeights := []float64{
		1, 2, -4,
		0, 0.25, 4,
	}
	if !reflect.DeepEqual(targetUpdatedWeights, layer.GetWeightsData()) {
		fmt.Println("--UPDATED WEIGTS--")
		fmt.Println(targetUpdatedWeights)
		fmt.Println(layer.GetWeightsData())
		t.Fatal()
	}
}

func TestDenseLayer_Backward_3(t *testing.T) {
	layer := layers.NewDenseLayer(1, 2)
	weights := []float64{
		-0.5, -0.75,
	}
	layer.LoadWeights(&weights)
	input := *mat.NewVecDense(1, []float64{
		4,
	})
	layer.Forward(&input)
	bias := []float64{
		1, 0,
	}
	layer.LoadBias(&bias)

	outGrads := layer.Backward(mat.NewVecDense(2, []float64{
		3, -2,
	}))
	targetWeightsGrads := mat.NewDense(2, 1, []float64{
		12, -8,
	})
	targetBiasGrads := mat.NewVecDense(2, []float64{
		3, -2,
	})
	targetOutGrads := mat.NewVecDense(1, []float64{
		0,
	})
	if !reflect.DeepEqual(targetOutGrads, outGrads) {
		fmt.Println("--OUT GRADS--")
		fmt.Println(targetOutGrads)
		fmt.Println(outGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(targetWeightsGrads, layer.GetOutWeightsGrads()) {
		fmt.Println("--WEIGHT GRADS--")
		fmt.Println(targetWeightsGrads)
		fmt.Println(layer.GetOutWeightsGrads())
		t.Fatal()
	}
	if !reflect.DeepEqual(targetBiasGrads, layer.GetOutBiasGrads()) {
		fmt.Println("--BIAS GRADS--")
		fmt.Println(targetBiasGrads)
		fmt.Println(layer.GetOutBiasGrads())
		t.Fatal()
	}

	learningRate := 0.25
	layer.ApplyGrads(&learningRate, layer.GetOutWeightsGrads(), layer.GetOutBiasGrads())
	targetUpdatedWeights := []float64{
		-3.5, 1.25,
	}
	if !reflect.DeepEqual(targetUpdatedWeights, layer.GetWeightsData()) {
		fmt.Println("--UPDATED WEIGTS--")
		fmt.Println(targetUpdatedWeights)
		fmt.Println(layer.GetWeightsData())
		t.Fatal()
	}
}
