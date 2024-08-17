package tests

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/layers"
)

func TestSumOfSquares_1(t *testing.T) {
	y := mat.NewVecDense(5, []float64{0.1, 0.2, 0.9, 0.7, 0.5})
	target := mat.NewVecDense(5, []float64{0, 0, 1, 1, 0})
	loss := layers.SumOfSquares(y, target)
	if target_loss := 0.4; loss != target_loss {
		fmt.Println(loss, target_loss)
		t.Fatal()
	}
}

func TestSumOfSquares_2(t *testing.T) {
	y := mat.NewVecDense(3, []float64{0.3, 0.3, 0})
	target := mat.NewVecDense(3, []float64{0, 1, 1})
	loss := layers.SumOfSquares(y, target)
	if target_loss := 1.58; loss != target_loss {
		fmt.Println(loss, target_loss)
		t.Fatal()
	}
}
