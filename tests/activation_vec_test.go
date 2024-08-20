package tests

import (
	"fmt"
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

func TestVReLU_Backward(t *testing.T) {
	layer := layers.NewVReLU()
	input1 := *mat.NewVecDense(4, []float64{
		1, -1, -888, 3,
	})
	layer.Forward(input1)
	inGrads1 := *mat.NewVecDense(4, []float64{
		1, 1, 1, 1,
	})
	output1 := layer.Backward(inGrads1)
	target1 := []float64{
		1, 0, 0, 1,
	}
	if !reflect.DeepEqual(output1.RawVector().Data, target1) {
		fmt.Println(output1.RawVector().Data)
		fmt.Println(target1)
		t.Fatal()
	}

	input2 := *mat.NewVecDense(4, []float64{-3, -4, -5, 1})
	layer.Forward(input2)
	inGrads2 := *mat.NewVecDense(4, []float64{
		1, 2, 3, 4,
	})
	output2 := layer.Backward(inGrads2)
	target2 := []float64{
		0, 0, 0, 4,
	}
	if !reflect.DeepEqual(output2.RawVector().Data, target2) {
		fmt.Println(output2.RawVector().Data)
		fmt.Println(target2)
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

func TestVLeakyReLU_Backward(t *testing.T) {
	layer := layers.NewVLeakyReLU(0.25)
	input1 := *mat.NewVecDense(4, []float64{
		1, -1, -888, 3,
	})
	layer.Forward(input1)
	inGrads1 := *mat.NewVecDense(4, []float64{
		1, 1, 1, 1,
	})
	output1 := layer.Backward(inGrads1)
	target1 := []float64{
		1, 0.25, 0.25, 1,
	}
	if !reflect.DeepEqual(output1.RawVector().Data, target1) {
		fmt.Println(output1.RawVector().Data)
		fmt.Println(target1)
		t.Fatal()
	}
	input2 := *mat.NewVecDense(4, []float64{-3, -4, -5, 1})
	layer.Forward(input2)
	inGrads2 := *mat.NewVecDense(4, []float64{
		1, 2, 3, 4,
	})
	output2 := layer.Backward(inGrads2)
	target2 := []float64{
		0.25, 0.50, 0.75, 4,
	}
	if !reflect.DeepEqual(output2.RawVector().Data, target2) {
		fmt.Println(output2.RawVector().Data)
		fmt.Println(target2)
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

func TestVELU_Backward(t *testing.T) {
	layer := layers.NewVELU(0.25)
	input1 := *mat.NewVecDense(4, []float64{
		1, -1, 0, 10,
	})
	layer.Forward(input1)
	inGrads1 := *mat.NewVecDense(4, []float64{
		1, 1, 1, 1,
	})
	output1 := layer.Backward(inGrads1)
	target1 := *mat.NewVecDense(4, []float64{
		1, 0.09196986, 0.25, 1,
	})
	if !functools.IsEqualVec(&target1, &output1, 0.001) {
		fmt.Println(output1.RawVector().Data)
		fmt.Println(target1)
		t.Fatal()
	}
	input2 := *mat.NewVecDense(4, []float64{-3, -4, -5, 1})
	layer.Forward(input2)
	inGrads2 := *mat.NewVecDense(4, []float64{
		1, 2, 3, 4,
	})
	output2 := layer.Backward(inGrads2)
	target2 := *mat.NewVecDense(4, []float64{
		0.012446767 * 1, 0.00457891 * 2, 0.001684487 * 3, 1 * 4,
	})
	if !functools.IsEqualVec(&target2, &output2, 0.001) {
		fmt.Println(output2.RawVector().Data)
		fmt.Println(target2)
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

func TestVSigmoid_Backward(t *testing.T) {
	layer := layers.NewVSigmoid()
	input1 := *mat.NewVecDense(4, []float64{
		-2, -1, 1, 2,
	})
	layer.Forward(input1)
	inGrads1 := *mat.NewVecDense(4, []float64{
		1, 1, 1, 1,
	})
	output1 := layer.Backward(inGrads1)
	target1 := *mat.NewVecDense(4, []float64{
		0.104993585, 0.196611933, 0.196611933, 0.104993585,
	})
	if !functools.IsEqualVec(&output1, &target1, 0.001) {
		fmt.Println(output1.RawVector().Data)
		fmt.Println(target1)
		t.Fatal()
	}
	input2 := *mat.NewVecDense(4, []float64{
		-2000, 2000, 0, 5.999,
	})
	layer.Forward(input2)
	inGrads2 := *mat.NewVecDense(4, []float64{
		1, 2, 3, 4,
	})
	output2 := layer.Backward(inGrads2)
	target2 := *mat.NewVecDense(4, []float64{
		0, 0, 0.75, 0.002468965 * 4,
	})
	if !functools.IsEqualVec(&output2, &target2, 0.001) {
		fmt.Println(output2.RawVector().Data)
		fmt.Println(target2)
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

func TestVTanh_Backward(t *testing.T) {
	layer := layers.NewVTanh()
	input1 := *mat.NewVecDense(4, []float64{
		-2, -1, 1, 2,
	})
	layer.Forward(input1)
	inGrads1 := *mat.NewVecDense(4, []float64{
		1, 1, 1, 1,
	})
	output1 := layer.Backward(inGrads1)
	target1 := *mat.NewVecDense(4, []float64{
		0.070650825, 0.419974342, 0.419974342, 0.070650825,
	})
	if !functools.IsEqualVec(&output1, &target1, 0.001) {
		fmt.Println(output1.RawVector().Data)
		fmt.Println(target1)
		t.Fatal()
	}
	input2 := *mat.NewVecDense(4, []float64{
		-2000, 2000, 0, 5.999,
	})
	layer.Forward(input2)
	inGrads2 := *mat.NewVecDense(4, []float64{
		1, 2, 3, 4,
	})
	output2 := layer.Backward(inGrads2)
	target2 := *mat.NewVecDense(4, []float64{
		0, 0, 3, 0.000024626 * 4,
	})
	if !functools.IsEqualVec(&output2, &target2, 0.001) {
		fmt.Println(output2.RawVector().Data)
		fmt.Println(target2)
		t.Fatal()
	}
}
