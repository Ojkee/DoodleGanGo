package conv_test

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
)

func TestMaxPool_1(t *testing.T) {
	layer := conv.NewMaxPool([2]int{2, 2}, [2]int{4, 4}, [2]int{2, 2}, 1)
	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			3, 1, 5, 4,
			2, 4, 2, -1,
			2, 7, 2, 0,
			7, 6, 1, -9,
		}),
	}
	layer.Forward(&input)
	targetFlat := []float64{
		4, 5, 7, 2,
	}
	if !reflect.DeepEqual(targetFlat, *layer.FlatOutput()) {
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
}

func TestMaxPool_2(t *testing.T) {
	layer := conv.NewMaxPool([2]int{2, 2}, [2]int{4, 4}, [2]int{2, 2}, 2)
	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			3, 1, 5, 4,
			2, 4, 2, -1,
			2, 7, 2, 0,
			7, 6, 1, -9,
		}),
		*mat.NewDense(4, 4, []float64{
			8, 2, 5, 0,
			4, 1, 5, 4,
			0, -3, 2, 2,
			-4, 0, 2, 3,
		}),
	}
	layer.Forward(&input)
	targetFlat := []float64{
		4, 5, 7, 2, 8, 5, 0, 3,
	}
	targetDeflat := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			4, 5, 7, 2,
		}),
		*mat.NewDense(2, 2, []float64{
			8, 5, 0, 3,
		}),
	}
	if !reflect.DeepEqual(targetFlat, *layer.FlatOutput()) {
		fmt.Println(*layer.FlatOutput())
		t.Fatal()
	}
	if !functools.IsEqualMatSlice(&targetDeflat, layer.DeflatOutput(), 0.001) {
		fmt.Println(*layer.DeflatOutput())
		t.Fatal()
	}
}

func TestMaxPool_3(t *testing.T) {
	layer := conv.NewMaxPool([2]int{2, 2}, [2]int{3, 3}, [2]int{2, 2}, 1)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			1, 2, 4,
			3, 2, 4,
			4, 4, 4,
		}),
	}
	layer.Forward(&input)
	targetFlat := []float64{
		3,
	}
	if !reflect.DeepEqual(targetFlat, *layer.FlatOutput()) {
		t.Fatal()
	}
}

func TestMaxPool_Backward_1(t *testing.T) {
	layer := conv.NewMaxPool([2]int{2, 2}, [2]int{4, 4}, [2]int{2, 2}, 2)
	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			0, 0, 0, 1,
			1, 0, 2, 0,
			2, 2, 2, 3,
			-2, 1, 3, 1,
		}),
		*mat.NewDense(4, 4, []float64{
			2, 1, 2, 1,
			1, 2, 1, 3,
			0, -2, 2, 2,
			-3, -4, 2, 2,
		}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{4, 2, -2, 3}),
		*mat.NewDense(2, 2, []float64{3, 4, 1, -5}),
	}
	result := layer.Backward(&inGrads)
	target := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			0, 0, 0, 0,
			4, 0, 2, 0,
			-2, 0, 0, 3,
			0, 0, 0, 0,
		}),
		*mat.NewDense(4, 4, []float64{
			3, 0, 0, 0,
			0, 0, 0, 4,
			1, 0, -5, 0,
			0, 0, 0, 0,
		}),
	}
	if !functools.IsEqualMatSlice(&target, result, 0.001) {
		functools.PrintMatSlice(&target, 1)
		functools.PrintMatSlice(result, 1)
		t.Fail()
	}
}

func TestMaxPool_Backward_2(t *testing.T) {
	layer := conv.NewMaxPool([2]int{3, 3}, [2]int{4, 4}, [2]int{3, 3}, 2)
	input := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			2, 1, 5, -2,
			1, 0, 3, 1,
			0, 4, 1, -2,
			4, 1, 2, 1,
		}),
		*mat.NewDense(4, 4, []float64{
			1, 3, 1, 2,
			0, 3, 4, 2,
			9, 2, -2, 1,
			3, 0, 2, 1,
		}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(1, 1, []float64{4}),
		*mat.NewDense(1, 1, []float64{-1}),
	}
	result := layer.Backward(&inGrads)
	target := []mat.Dense{
		*mat.NewDense(4, 4, []float64{
			0, 0, 4, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
		}),
		*mat.NewDense(4, 4, []float64{
			0, 0, 0, 0,
			0, 0, 0, 0,
			-1, 0, 0, 0,
			0, 0, 0, 0,
		}),
	}
	if !functools.IsEqualMatSlice(&target, result, 0.001) {
		functools.PrintMatSlice(&target, 1)
		functools.PrintMatSlice(result, 1)
		t.Fail()
	}
}
