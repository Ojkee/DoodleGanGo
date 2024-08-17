package layers

import "gonum.org/v1/gonum/mat"

func SumOfSquares(input *mat.VecDense, targets *mat.VecDense) float64 {
	var rvd mat.VecDense
	rvd.SubVec(input, targets)
	var r mat.VecDense
	r.MulElemVec(rvd.SliceVec(0, rvd.Len()), rvd.SliceVec(0, rvd.Len()))
	result := 0.0
	for i := range rvd.RawVector().Data {
		result += r.RawVector().Data[i]
	}
	return result
}
