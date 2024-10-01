package losses

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type BatchSize struct {
	batchSizeInt   int
	batchSizeFloat float64
}

type OutputLen struct {
	outputLenInt   int
	outputLenFloat float64
}

type Loss interface {
	CalculateAvg(yHat, y *[]mat.VecDense) float64
	CalculateTotal(yHat, y *[]mat.VecDense) float64
}

func SumOfSquaresBatch(yHat, y *[]mat.VecDense, batchSizeInt, outputLen *int) float64 {
	retSum := 0.0
	for i := range *batchSizeInt {
		retSum += SumOfSquares(&(*yHat)[i], &(*y)[i], outputLen)
	}
	return retSum
}

func SumOfSquares(yHat, y *mat.VecDense, outputLen *int) float64 {
	retSum := 0.0
	for j := range *outputLen {
		label := (*yHat).AtVec(j)
		pred := (*y).AtVec(j)
		retSum += math.Pow(label-pred, 2)
	}
	return retSum
}
