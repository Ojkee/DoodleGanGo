package losses

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ResidualSumOfSquares struct {
	BatchSize
	OutputLen
}

func NewResidualSumOfSquares(batchSize, outputLen int) ResidualSumOfSquares {
	return ResidualSumOfSquares{
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

func (loss *ResidualSumOfSquares) CalculateAvg(yHat, y *[]mat.VecDense) float64 {
	return loss.CalculateTotal(yHat, y) / loss.batchSizeFloat
}

func (loss *ResidualSumOfSquares) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	retSum := float64(0.0)
	for i := range loss.batchSizeInt {
		for j := range loss.outputLenInt {
			label := (*yHat)[i].AtVec(j)
			pred := (*y)[i].AtVec(j)
			retSum += math.Pow(label-pred, 2)
		}
	}
	return retSum
}
