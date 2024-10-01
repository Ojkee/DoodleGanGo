package losses

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type RootMeanSquareError struct {
	BatchSize
	OutputLen
}

func NewRootMeanSquareError(batchSize, outputLen int) RootMeanSquareError {
	return RootMeanSquareError{
		BatchSize: BatchSize{
			batchSizeInt:   batchSize,
			batchSizeFloat: float64(batchSize),
		},
		OutputLen: OutputLen{
			outputLenInt:   outputLen,
			outputLenFloat: float64(outputLen),
		},
	}
}

func (loss *RootMeanSquareError) CalculateAvg(yHat, y *[]mat.VecDense) float64 {
	return loss.CalculateTotal(yHat, y) / loss.batchSizeFloat
}

func (loss *RootMeanSquareError) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	retVal := 0.0
	for i := range loss.batchSizeInt {
		retVal += math.Sqrt(
			SumOfSquares(&(*yHat)[i], &(*y)[i], &loss.outputLenInt) / loss.outputLenFloat,
		)
	}
	return retVal
}
