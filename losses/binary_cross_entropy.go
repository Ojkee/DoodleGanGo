package losses

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type BinaryCrossEntropy struct {
	BatchSize
}

func NewBinaryCrossEntropy(batchSize int) BinaryCrossEntropy {
	return BinaryCrossEntropy{
		BatchSize{
			batchSizeInt:   batchSize,
			batchSizeFloat: float64(batchSize),
		},
	}
}

func (loss *BinaryCrossEntropy) CalculateAvg(yHat, y *[]mat.VecDense) float64 {
	return loss.CalculateTotal(yHat, y) / loss.batchSizeFloat
}

func (loss *BinaryCrossEntropy) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	retVal := 0.0
	for i := range loss.batchSizeInt {
		y_ := (*y)[i].AtVec(0)
		yHat_ := (*yHat)[i].AtVec(0)
		if yHat_ == 0 && y_ == 1 {
			yHat_ = 10e-8
		} else if yHat_ == 1 && y_ == 0 {
			yHat_ = 0.99999999
		}
		retVal += y_*math.Log(yHat_) + (1-y_)*math.Log(1-yHat_)
	}
	return -retVal
}
