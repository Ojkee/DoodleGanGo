package losses

import "gonum.org/v1/gonum/mat"

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
	return 0
}

func (loss *BinaryCrossEntropy) CalculateTotal(yHat, y *[]mat.VecDense) float64 {
	retSum := float64(0.0)
	return retSum
}
