package losses

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type CrossEntropy struct {
	BatchSize
	outputClasses int
}

func NewCrossEntropy(batchSize, outputLen int) CrossEntropy {
	return CrossEntropy{
		BatchSize: BatchSize{
			batchSizeInt:   batchSize,
			batchSizeFloat: float64(batchSize),
		},
		outputClasses: outputLen,
	}
}

func (loss *CrossEntropy) CalculateAvg(yHat, y *[]mat.VecDense) float64 {
	return loss.CalculateTotal(yHat, y) / loss.batchSizeFloat
}

func (loss *CrossEntropy) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	retSum := float64(0.0)
	for i := range loss.batchSizeInt {
		for j := range loss.outputClasses {
			label := (*y)[i].AtVec(j)
			predLog := math.Log((*yHat)[i].AtVec(j))
			retSum -= label * predLog
		}
	}
	return retSum
}
