package losses

import (
	"gonum.org/v1/gonum/mat"
)

type MeanSquareError struct {
	BatchSize
	OutputLen
}

func NewMeanSquareError(batchSize, outputLen int) MeanSquareError {
	return MeanSquareError{
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

func (loss *MeanSquareError) CalculateAvg(yHat, y *[]mat.VecDense) float64 {
	return SumOfSquaresBatch(
		yHat,
		y,
		&loss.batchSizeInt,
		&loss.outputLenInt,
	) / (loss.outputLenFloat * loss.batchSizeFloat)
}

func (loss *MeanSquareError) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	return SumOfSquaresBatch(yHat, y, &loss.batchSizeInt, &loss.outputLenInt) / loss.outputLenFloat
}
