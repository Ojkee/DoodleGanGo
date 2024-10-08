package losses_test

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
	"DoodleGan/losses"
)

func TestMSE_1(t *testing.T) {
	loss := losses.NewMeanSquareError(1, 3)
	yHat := []mat.VecDense{
		*mat.NewVecDense(3, []float64{0.57, 0.20, 0.23}),
	}
	y := []mat.VecDense{
		*mat.NewVecDense(3, []float64{1, 0, 0}),
	}
	result := loss.CalculateAvg(&yHat, &y)
	resultTotal := loss.CalculateTotal(&yHat, &y)
	target := float64(0.0926)
	targetTotal := float64(0.0926)
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

func TestMSE_2(t *testing.T) {
	loss := losses.NewMeanSquareError(3, 3)
	yHat := []mat.VecDense{
		*mat.NewVecDense(3, []float64{0.57, 0.20, 0.23}),
		*mat.NewVecDense(3, []float64{0.22, 0.20, 0.58}),
		*mat.NewVecDense(3, []float64{0.24, 0.52, 0.24}),
	}
	y := []mat.VecDense{
		*mat.NewVecDense(3, []float64{1, 0, 0}),
		*mat.NewVecDense(3, []float64{0, 0, 1}),
		*mat.NewVecDense(3, []float64{0, 1, 0}),
	}
	result := loss.CalculateAvg(&yHat, &y)
	resultTotal := loss.CalculateTotal(&yHat, &y)
	target := float64(0.09868)
	targetTotal := float64(0.29606)
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
