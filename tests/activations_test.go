package tests

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/layers"
)

func TestReLU(t *testing.T) {
	layer := layers.NewReLU()
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

func TestLeakyReLU(t *testing.T) {
	layer := layers.NewLeakyReLU(0.1)
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, -888, 0}),
		*mat.NewDense(2, 2, []float64{-3, -4, -5, 2}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1.0 * 0.1, -888.0 * 0.1, 0}),
		*mat.NewDense(2, 2, []float64{-3.0 * 0.1, -4.0 * 0.1, -5.0 * 0.1, 2}),
	}
	if !layers.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestELU(t *testing.T) {
	layer := layers.NewELU(0.5)
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -1, 0, 10}),
		*mat.NewDense(2, 2, []float64{-2, 0.1, -10, -0.1}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{1, -0.316060279, 0, 10}),
		*mat.NewDense(2, 2, []float64{-0.432332358, 0.1, -0.4999773, -0.047581291}),
	}
	if !layers.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestSigmoid(t *testing.T) {
	layer := layers.NewSigmoid()
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{-2, -1, 1, 2}),
		*mat.NewDense(2, 2, []float64{-2000, 2000, 0, 5.999}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{0.119202922, 0.268941421, 0.731058579, 0.880797078}),
		*mat.NewDense(2, 2, []float64{0, 1, 0.5, 0.997524909}),
	}
	if !layers.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}

func TestTanh(t *testing.T) {
	layer := layers.NewTanh()
	output := layer.Forward([]mat.Dense{
		*mat.NewDense(2, 2, []float64{-2, -1, 1, 2}),
		*mat.NewDense(2, 2, []float64{-20, 2000, 0, 5.999}),
	})
	target := []mat.Dense{
		*mat.NewDense(2, 2, []float64{-0.96402758007, -0.76159415595, 0.76159415595, 0.96402758007}),
		*mat.NewDense(2, 2, []float64{-1, 1, 0, 0.99998768705}),
	}
	if !layers.IsEqualMat(&output, &target, 0.001) {
		t.Fatal()
	}
}
