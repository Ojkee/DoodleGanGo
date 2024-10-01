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
			pred := (*yHat)[i].AtVec(j)
			if pred == 0.0 && label == 1 {
				pred = 10e-8
			} else if pred == 1 && label == 0 {
				pred = 0.99999999
			}
			predLog := math.Log(pred)
			retSum -= label * predLog
		}
	}
	return retSum
}
