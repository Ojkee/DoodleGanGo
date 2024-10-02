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

func IsEqualVal(A, B *float64, eps float64) bool {
	if math.Abs(*A-*B) > eps {
		return false
	}
	return true
}

func RepeatSlice[T any](v T, n int) []T {
	result := make([]T, n)
	for i := range n {
		result[i] = v
	}
	return result
}

func ArgToSliceLabel(n, idx int) []float64 {
	retVal := make([]float64, n)
	retVal[idx] = 1.0
	return retVal
}

func VecToMatSlice(source *mat.VecDense, height, width int) []mat.Dense {
	channelPixels := height * width
	numChannels := len(source.RawVector().Data) / channelPixels
	retVal := make([]mat.Dense, numChannels)
	sourceData := source.RawVector().Data
	for i := range numChannels {
		retVal[i] = *mat.NewDense(
			height,
			width,
			sourceData[i*channelPixels:(i+1)*channelPixels],
		)
	}
	return retVal
}

func FlattenMat(A *mat.Dense) []float64 {
	m, _ := (*A).Dims()
	retVal := make([]float64, 0)
	for r := range m {
		retVal = append(retVal, (*A).RawRowView(r)...)
	}
	return retVal
}
