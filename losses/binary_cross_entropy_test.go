package losses_test

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
	"DoodleGan/losses"
)

func TestBinaryCrossEntropy_1(t *testing.T) {
	loss := losses.NewBinaryCrossEntropy(1)
	yHat := []mat.VecDense{
		*mat.NewVecDense(1, []float64{0.3}),
	}
	y := []mat.VecDense{
		*mat.NewVecDense(1, []float64{1}),
	}
	result := loss.CalculateAvg(&yHat, &y)
	resultTotal := loss.CalculateTotal(&yHat, &y)
	target := float64(1.20397)
	targetTotal := float64(1.20397)
	if !functools.IsEqualVal(&target, &result, 0.001) {
		fmt.Println(target)
		fmt.Println(result)
		t.Fail()
	}
	if !functools.IsEqualVal(&targetTotal, &resultTotal, 0.001) {
		fmt.Println(targetTotal)
		fmt.Println(resultTotal)
		t.Fail()
	}
}

func TestBinaryCrossEntropy_2(t *testing.T) {
	loss := losses.NewBinaryCrossEntropy(1)
	yHat := []mat.VecDense{
		*mat.NewVecDense(1, []float64{0.7}),
	}
	y := []mat.VecDense{
		*mat.NewVecDense(1, []float64{0}),
	}
	result := loss.CalculateAvg(&yHat, &y)
	resultTotal := loss.CalculateTotal(&yHat, &y)
	target := float64(1.20397)
	targetTotal := float64(1.20397)
	if !functools.IsEqualVal(&target, &result, 0.001) {
		fmt.Println(target)
		fmt.Println(result)
		t.Fail()
	}
	if !functools.IsEqualVal(&targetTotal, &resultTotal, 0.001) {
		fmt.Println(targetTotal)
		fmt.Println(resultTotal)
		t.Fail()
	}
}

func TestBinaryCrossEntropy_3(t *testing.T) {
	loss := losses.NewBinaryCrossEntropy(5)
	yHat := []mat.VecDense{
		*mat.NewVecDense(1, []float64{0.7}),
		*mat.NewVecDense(1, []float64{0.5}),
		*mat.NewVecDense(1, []float64{1}),
		*mat.NewVecDense(1, []float64{0}),
		*mat.NewVecDense(1, []float64{0}),
	}
	y := []mat.VecDense{
		*mat.NewVecDense(1, []float64{0}),
		*mat.NewVecDense(1, []float64{1}),
		*mat.NewVecDense(1, []float64{0}),
		*mat.NewVecDense(1, []float64{1}),
		*mat.NewVecDense(1, []float64{0}),
	}
	result := loss.CalculateAvg(&yHat, &y)
	resultTotal := loss.CalculateTotal(&yHat, &y)
	target := float64(7.74769)
	targetTotal := float64(38.73847)
	if !functools.IsEqualVal(&target, &result, 0.001) {
		fmt.Println(target)
		fmt.Println(result)
		t.Fail()
	}
	if !functools.IsEqualVal(&targetTotal, &resultTotal, 0.001) {
		fmt.Println(targetTotal)
		fmt.Println(resultTotal)
		t.Fail()
	}
}
