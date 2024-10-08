package optimizers_test

import (
	"testing"

	"DoodleGan/conv"
)

func TestAdam_zero_momentum_zero_rho(t *testing.T) {
	conv1 := conv.NewConv2D([2]int{2, 1}, 2, [2]int{2, 2}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})
	// act1 := conv.NewReLU()
	conv2 := conv.NewConv2D([2]int{1, 2}, 1, [2]int{1, 2}, 2, [2]int{1, 1}, [4]int{0, 0, 0, 0})

	filter1 := []float64{
		1, -2, -1, 2,
		2, -1, 2, 1,
	}
	bias1 := []float64{
		1, -1,
	}
	conv1.LoadFilter(&filter1)
	conv1.LoadBias(&bias1)

	filter2 := []float64{
		3, 1, -2, 2,
	}
	bias2 := []float64{
		2,
	}
	conv2.LoadFilter(&filter2)
	conv2.LoadBias(&bias2)
}
