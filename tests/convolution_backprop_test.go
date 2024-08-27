package tests

import (
	"fmt"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
)

func TestConv2D_Backward_1(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2,
		-2, 0.5,
	}
	layer.LoadFilter(&filter)
	bias := []float64{1}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			1, 2, 1,
			0, 3, -2,
			3, 4, 1,
		}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			3, 2,
			1, -2.5,
		}),
	}

	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			3, 8, 4,
			-5, -3, -4,
			-2, 5.5, -1.25,
		}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			-0.5, 16,
			-1, 6.5,
		}),
	}
	targetBiasGrads := []float64{
		3.5,
	}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(outGrads)
		fmt.Println(targetOutGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fatal()
	}
}

func TestConv2D_Backward_2(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 1, [2]int{3, 3}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2,
		-2, 0.5,

		3, 1,
		2, -3,
	}
	layer.LoadFilter(&filter)
	bias := []float64{1}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			1, 2, 1,
			0, 3, -2,
			3, 4, 1,
		}),
		*mat.NewDense(3, 3, []float64{
			2, 3, 2,
			1, -2, 1,
			2, 4, -1,
		}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			3, 2,
			1, -2.5,
		}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			3, 8, 4,
			-5, -3, -4,
			-2, 5.5, -1.25,
		}),
		*mat.NewDense(3, 3, []float64{
			9, 9, 2,
			9, -11.5, -8.5,
			2, -8, 7.5,
		}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			-0.5, 16,
			-1, 6.5,
		}),
		*mat.NewDense(2, 2, []float64{
			18, 8.5,
			-9, 2.5,
		}),
	}
	targetBiasGrads := []float64{
		3.5,
	}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(*outGrads)
		fmt.Println(targetOutGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fatal()
	}
}

func TestConv2D_Backward_3(t *testing.T) {
	layer := conv.NewConv2D([2]int{2, 2}, 2, [2]int{3, 3}, 1, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	filter := []float64{
		1, 2,
		-2, 0.5,

		3, 1,
		2, -3,
	}
	layer.LoadFilter(&filter)
	bias := []float64{
		1, 2,
	}
	layer.LoadBias(&bias)
	input := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			1, 2, 1,
			0, 3, -2,
			3, 4, 1,
		}),
	}
	layer.Forward(&input)
	inGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			3, 2,
			1, -2.5,
		}),
		*mat.NewDense(2, 2, []float64{
			2, 5,
			-1, 2,
		}),
	}
	layer.Backward(&inGrads)

	outGrads := layer.DeflatOutGrads()
	filterGrads := layer.GetFilterGrads()
	biasGrads := layer.GetBiasGrads()

	targetOutGrads := []mat.Dense{
		*mat.NewDense(3, 3, []float64{
			9, 25, 9,
			-4, 6, -17,
			-4, 12.5, -7.25,
		}),
	}
	targetFilterGrads := []mat.Dense{
		*mat.NewDense(2, 2, []float64{
			-0.5, 16,
			-1, 6.5,
		}),
		*mat.NewDense(2, 2, []float64{
			18, 2,
			20, -6,
		}),
	}
	targetBiasGrads := []float64{
		3.5,
		8,
	}

	if !reflect.DeepEqual(*filterGrads, targetFilterGrads) {
		fmt.Println("== FILTER GRADS ==")
		fmt.Println(*filterGrads)
		fmt.Println(targetFilterGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(*outGrads, targetOutGrads) {
		fmt.Println("== OUT GRADS ==")
		fmt.Println(outGrads)
		fmt.Println(targetOutGrads)
		t.Fatal()
	}
	if !reflect.DeepEqual(*biasGrads, targetBiasGrads) {
		fmt.Println("== BIAS GRADS ==")
		fmt.Println(biasGrads)
		fmt.Println(targetBiasGrads)
		t.Fatal()
	}
}
