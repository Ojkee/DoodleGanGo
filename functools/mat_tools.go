package functools

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func PrintMatArray(A *[]mat.Dense, precision int) {
	precision = max(0, precision)
	precision = min(52, precision)
	for i := range len(*A) {
		PrintMat(&(*A)[i], precision)
		fmt.Println()
	}
}

func PrintMat(A *mat.Dense, precision int) {
	h, w := (*A).Dims()
	for j := range h {
		for k := range w {
			value := (*A).At(j, k)
			fmt.Printf("%."+fmt.Sprint(precision)+"f ", value)
		}
		fmt.Println()
	}
}

func FlattenMat(A *mat.Dense) []float64 {
	m, _ := (*A).Dims()
	result := make([]float64, 0)
	for r := range m {
		result = append(result, (*A).RawRowView(r)...)
	}
	return result
}

func IsEqualMat(A, B *[]mat.Dense, eps float64) bool {
	if len(*A) != len(*B) {
		return false
	}
	for d := range *A {
		var res mat.Dense
		res.Sub(&(*A)[d], &(*B)[d])
		m, n := res.Dims()
		for i := range m {
			for j := range n {
				if math.Abs(res.At(i, j)) > eps {
					return false
				}
			}
		}
	}
	return true
}

func IsEqualVec(A, B *mat.VecDense, eps float64) bool {
	if A.Len() != B.Len() {
		return false
	}
	for i := range A.Len() {
		if math.Abs(A.RawVector().Data[i]-B.RawVector().Data[i]) > eps {
			return false
		}
	}
	return true
}

func IsEqual(A, B *[]float64, eps float64) bool {
	if len(*A) != len(*B) {
		return false
	}
	for i := range *A {
		if math.Abs((*A)[i]-(*B)[i]) > eps {
			return false
		}
	}
	return true
}
