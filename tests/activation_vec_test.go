package tests

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
	"DoodleGan/layers"
)

func TestSoftmax(t *testing.T) {
	layer := layers.NewSoftmax()
	outputVec := layer.Forward(*mat.NewVecDense(4, []float64{
		1.1, 2.2, 0.2, -1.7,
	}))
	output := outputVec.RawVector().Data
	target := []float64{
		0.224, 0.672, 0.091, 0.013,
	}
	if !functools.IsEqual(&target, &output, 0.001) {
		t.Fatal()
	}
}

func TestVReLU(t *testing.T) {
	layer := layers.NewVReLU()
	output1 := layer.Forward(*mat.NewVecDense(4, []float64{1, -1, -888, 0}))
	target1 := *mat.NewVecDense(4, []float64{1, 0, 0, 0})
	if !reflect.DeepEqual(output1, target1) {
		t.Fatal()
	}
	output2 := layer.Forward(*mat.NewVecDense(4, []float64{-3, -4, -5, 1}))
	target2 := *mat.NewVecDense(4, []float64{0, 0, 0, 1})
	if !reflect.DeepEqual(output2, target2) {
		t.Fatal()
	}
}

func TestVLeakyReLU(t *testing.T) {
	layer := layers.NewVLeakyReLU(0.1)
	output1 := layer.Forward(*mat.NewVecDense(4, []float64{1, -1, -888, 0}))
	target1 := *mat.NewVecDense(4, []float64{1, -1.0 * 0.1, -888.0 * 0.1, 0})
	if !functools.IsEqualVec(&output1, &target1, 0.001) {
		t.Fatal()
	}
	output2 := layer.Forward(*mat.NewVecDense(4, []float64{-3, -4, -5, 2}))
	target2 := *mat.NewVecDense(4, []float64{-3.0 * 0.1, -4.0 * 0.1, -5.0 * 0.1, 2})
	if !functools.IsEqualVec(&output2, &target2, 0.001) {
		t.Fatal()
	}
}

func TestVELU(t *testing.T) {
	layer := layers.NewVELU(0.5)
	output1 := layer.Forward(*mat.NewVecDense(4, []float64{1, -1, 0, 10}))
	target1 := *mat.NewVecDense(4, []float64{1, -0.316060279, 0, 10})
	if !functools.IsEqualVec(&output1, &target1, 0.001) {
		t.Fatal()
	}
	output2 := layer.Forward(*mat.NewVecDense(4, []float64{-2, 0.1, -10, -0.1}))
	target2 := *mat.NewVecDense(4, []float64{-0.432332358, 0.1, -0.4999773, -0.047581291})
	if !functools.IsEqualVec(&output2, &target2, 0.001) {
		t.Fatal()
	}
}

func TestVSigmoid(t *testing.T) {
	layer := layers.NewVSigmoid()
	output1 := layer.Forward(*mat.NewVecDense(4, []float64{-2, -1, 1, 2}))
	target1 := *mat.NewVecDense(4, []float64{0.119202922, 0.268941421, 0.731058579, 0.880797078})
	if !functools.IsEqualVec(&output1, &target1, 0.001) {
		t.Fatal()
	}
	output2 := layer.Forward(*mat.NewVecDense(4, []float64{-2000, 2000, 0, 5.999}))
	target2 := *mat.NewVecDense(4, []float64{0, 1, 0.5, 0.997524909})
	if !functools.IsEqualVec(&output2, &target2, 0.001) {
		t.Fatal()
	}
}

func TestVTanh(t *testing.T) {
	layer := layers.NewVTanh()
	output1 := layer.Forward(*mat.NewVecDense(4, []float64{-2, -1, 1, 2}))
	target1 := *mat.NewVecDense(4, []float64{-0.96402758007, -0.76159415595, 0.76159415595, 0.96402758007})
	if !functools.IsEqualVec(&output1, &target1, 0.001) {
		t.Fatal()
	}
	output2 := layer.Forward(*mat.NewVecDense(4, []float64{-20, 2000, 0, 5.999}))
	target2 := *mat.NewVecDense(4, []float64{-1, 1, 0, 0.99998768705})
	if !functools.IsEqualVec(&output2, &target2, 0.001) {
		t.Fatal()
	}
}
