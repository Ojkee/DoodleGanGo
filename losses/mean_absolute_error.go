package losses

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type MeanAbsoluteError struct {
	BatchSize
	OutputLen
}

func NewMeanAbsoluteError(batchSize, outputLen int) MeanAbsoluteError {
	return MeanAbsoluteError{
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

func (loss *MeanAbsoluteError) CalculateAvg(yHat, y *[]mat.VecDense) float64 {
	return loss.sumOfAbs(yHat, y) / (loss.batchSizeFloat * loss.outputLenFloat)
}

func (loss *MeanAbsoluteError) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	return loss.sumOfAbs(yHat, y) / loss.outputLenFloat
}

func (loss *MeanAbsoluteError) sumOfAbs(yHat, y *[]mat.VecDense) float64 {
	retVal := 0.0
	for i := range loss.batchSizeInt {
		for j := range loss.outputLenInt {
			label := (*y)[i].AtVec(j)
			pred := (*yHat)[i].AtVec(j)
			retVal += math.Abs(label - pred)
		}
	}
	return retVal
}
