package tests

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
)

func TestReLU(t *testing.T) {
	layer := conv.NewReLU()
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, -888, 0}),
		*mat.NewDense(2, 2, []float64{-3, -4, -5, 1}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 0, 0, 0}),
		*mat.NewDense(2, 2, []float64{0, 0, 0, 1}),
	}
	if !reflect.DeepEqual(output, target) {
		t.Fatal()
	}
}

func TestReLU_Backward(t *testing.T) {
	layer := conv.NewReLU()
	input := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, -888, 0}),
		*mat.NewDense(2, 2, []float64{-3, -4, -5, 1}),
	}
	layer.Forward(input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		*mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
	}
	output := layer.Backward(inGrads)
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 0, 0, 0}),
		*mat.NewDense(2, 2, []float64{0, 0, 0, 4}),
	}
	if !reflect.DeepEqual(output, target) {
		fmt.Println(target)
		fmt.Println(output)
		t.Fatal()
	}
}

func TestLeakyReLU(t *testing.T) {
	layer := conv.NewLeakyReLU(0.1)
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, -888, 0}),
		*mat.NewDense(2, 2, []float64{-3, -4, -5, 2}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1.0 * 0.1, -888.0 * 0.1, 0}),
		*mat.NewDense(2, 2, []float64{-3.0 * 0.1, -4.0 * 0.1, -5.0 * 0.1, 2}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestLeakyReLU_Backward(t *testing.T) {
	alpha := 0.1
	layer := conv.NewLeakyReLU(alpha)
	layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, -888, 0}),
		*mat.NewDense(2, 2, []float64{-3, -4, -5, 2}),
	})
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		*mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
	}
	output := layer.Backward(inGrads)
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 0.1, 0.1, 0.1}),
		*mat.NewDense(2, 2, []float64{0.1, 0.2, 0.3, 4}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		fmt.Println(output)
		fmt.Println(target)
		t.Fatal()
	}
}

func TestELU(t *testing.T) {
	layer := conv.NewELU(0.5)
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, 0, 10}),
		*mat.NewDense(2, 2, []float64{-2, 0.1, -10, -0.1}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -0.316060279, 0, 10}),
		*mat.NewDense(2, 2, []float64{-0.432332358, 0.1, -0.4999773, -0.047581291}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestELU_Backward(t *testing.T) {
	alpha := 0.25
	layer := conv.NewELU(alpha)
	input := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, 0, 10}),
		*mat.NewDense(2, 2, []float64{-3, -4, -5, 1}),
	}
	layer.Forward(input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		*mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
	}
	output := layer.Backward(inGrads)
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 0.09196986, 0.25, 1}),
		*mat.NewDense(2, 2, []float64{0.012446767 * 1, 0.00457891 * 2, 0.001684487 * 3, 1 * 4}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		fmt.Println(output)
		fmt.Println(target)
		t.Fatal()
	}
}

func TestSigmoid(t *testing.T) {
	layer := conv.NewSigmoid()
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{-2, -1, 1, 2}),
		*mat.NewDense(2, 2, []float64{-2000, 2000, 0, 5.999}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.119202922, 0.268941421, 0.731058579, 0.880797078}),
		*mat.NewDense(2, 2, []float64{0, 1, 0.5, 0.997524909}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestSigmoid_Backward(t *testing.T) {
	layer := conv.NewSigmoid()
	input1 := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-2, -1, 1, 2}),
		*mat.NewDense(2, 2, []float64{-2000, 2000, 0, 5.999}),
	}
	layer.Forward(input1)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		*mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
	}
	output := layer.Backward(inGrads)
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.104993585, 0.196611933, 0.196611933, 0.104993585}),
		*mat.NewDense(2, 2, []float64{0, 0, 0.75, 0.002468965 * 4}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		fmt.Println(output)
		fmt.Println(target)
		t.Fatal()
	}
}

func TestTanh(t *testing.T) {
	layer := conv.NewTanh()
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{-2, -1, 1, 2}),
		*mat.NewDense(2, 2, []float64{-20, 2000, 0, 5.999}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-0.96402758007, -0.76159415595, 0.76159415595, 0.96402758007}),
		*mat.NewDense(2, 2, []float64{-1, 1, 0, 0.99998768705}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestTanh_Backward(t *testing.T) {
	layer := conv.NewTanh()
	input := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-2, -1, 1, 2}),
		*mat.NewDense(2, 2, []float64{-2000, 2000, 0, 5.999}),
	}
	layer.Forward(input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
		*mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
	}
	output := layer.Backward(inGrads)
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.070650825, 0.419974342, 0.419974342, 0.070650825}),
		*mat.NewDense(2, 2, []float64{0, 0, 3, 0.000024626 * 4}),
	}
	if !functools.IsEqualMat(&output, &target, 0.001) {
		fmt.Println(output)
		fmt.Println(target)
		t.Fatal()
	}
}
