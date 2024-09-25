package tests

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
)

func TestPool_1(t *testing.T) {
	layer := conv.NewAvgPool([2]int{2, 2}, [2]int{4, 4}, [2]int{2, 2})
	input := []mat.Dense{*mat.NewDense(4, 4, []float64{
		1, 2, -1, -2,
		3, 5, 5, 1,
		-3, 5, 1, 2,
		-2, 1, 2, 8,
	})}
	layer.Forward(&input)
	target := []float64{
		11. / 4, 3. / 4,
		1. / 4, 13. / 4,
	}
	if !functools.IsEqual(&target, layer.FlatOutput(), 0.01) {
		fmt.Println(target)
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
}

func TestPool_2(t *testing.T) {
	layer := conv.NewAvgPool([2]int{2, 2}, [2]int{5, 5}, [2]int{2, 2})
	input := []mat.Dense{*mat.NewDense(5, 5, []float64{
		1, 2, -1, -2, 5,
		3, 5, 5, 1, 0,
		-3, 5, 1, 2, 2,
		-2, 1, 2, 8, 7,
		2, 3, 5, 1, 0,
	})}
	layer.Forward(&input)
	target := []float64{
		11. / 4, 3. / 4,
		1. / 4, 13. / 4,
	}
	if !functools.IsEqual(&target, layer.FlatOutput(), 0.01) {
		fmt.Println(target)
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
}

func TestPool_3(t *testing.T) {
	layer := conv.NewAvgPool([2]int{3, 3}, [2]int{5, 5}, [2]int{3, 3})
	input := []mat.Dense{
		*mat.NewDense(5, 5, []float64{
			1, 2, -1, -2, 5,
			3, 5, 5, 1, 0,
			-3, 5, 1, 2, 2,
			-2, 1, 2, 8, 7,
			2, 3, 5, 1, 0,
		}),
		*mat.NewDense(5, 5, []float64{
			5, 3, 9, 0, 1,
			0, -2, 0, 9, -3,
			-4, 2, 1, 9, 3,
			5, 3, 2, 1, 1,
			0, -3, -3, -4, 5,
		}),
	}
	layer.Forward(&input)
	targetFlat := []float64{
		18. / 9,
		14. / 9,
	}
	targetDeflat := []mat.Dense{
		*mat.NewDense(1, 1, []float64{18. / 9}),
		*mat.NewDense(1, 1, []float64{14. / 9}),
	}
	if !functools.IsEqual(&targetFlat, layer.FlatOutput(), 0.01) {
		fmt.Println(targetFlat)
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
	if !functools.IsEqualMat(&targetDeflat, layer.DeflatOutput(), 0.01) {
		fmt.Println(targetDeflat)
		fmt.Println(layer.DeflatOutput())
		t.Fatal()
	}
}

func TestPool_Backward_1(t *testing.T) {
	layer := conv.NewAvgPool([2]int{2, 2}, [2]int{4, 4}, [2]int{2, 2})
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 2, 4, -1}),
	}
	result := layer.Backward(&inGrads)
	target := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 2.0,
			1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 2.0,
			1.0, 1.0, -1.0 / 4.0, -1.0 / 4.0,
			1.0, 1.0, -1.0 / 4.0, -1.0 / 4.0,
		}),
	}
	if !functools.IsEqualMat(&target, result, 0.001) {
		functools.PrintMatArray(&target, 2)
		functools.PrintMatArray(result, 2)
		t.Fail()
	}
}

func TestPool_Backward_2(t *testing.T) {
	layer := conv.NewAvgPool([2]int{2, 2}, [2]int{5, 5}, [2]int{2, 2})
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 2, 4, -1}),
	}
	result := layer.Backward(&inGrads)
	target := []mat.Dense{
		*mat.NewDense(5, 5, []float64{
			1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 2.0, 0.0,
			1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 2.0, 0.0,
			1.0, 1.0, -1.0 / 4.0, -1.0 / 4.0, 0.0,
			1.0, 1.0, -1.0 / 4.0, -1.0 / 4.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0,
		}),
	}
	if !functools.IsEqualMat(&target, result, 0.001) {
		functools.PrintMatArray(&target, 2)
		functools.PrintMatArray(result, 2)
		t.Fail()
	}
}

func TestPool_Backward_3(t *testing.T) {
	layer := conv.NewAvgPool([2]int{3, 3}, [2]int{4, 4}, [2]int{3, 3})
	inGrads := []mat.Dense{
		*mat.NewDense(1, 1, []float64{3}),
		*mat.NewDense(1, 1, []float64{18}),
	}
	result := layer.Backward(&inGrads)
	target := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			3.0 / 9.0, 3.0 / 9.0, 3.0 / 9.0, 0,
			3.0 / 9.0, 3.0 / 9.0, 3.0 / 9.0, 0,
			3.0 / 9.0, 3.0 / 9.0, 3.0 / 9.0, 0,
			0, 0, 0, 0,
		}),
		*mat.NewDense(4, 4, []float64{
			2, 2, 2, 0,
			2, 2, 2, 0,
			2, 2, 2, 0,
			0, 0, 0, 0,
		}),
	}
	if !functools.IsEqualMat(&target, result, 0.001) {
		functools.PrintMatArray(&target, 2)
		functools.PrintMatArray(result, 2)
		t.Fail()
	}
}
