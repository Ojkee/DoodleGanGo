package conv

import "gonum.org/v1/gonum/mat"

type AvgPool struct {
	poolSize   MatSize
	stride     Stride
	matPool    mat.Dense
	outputSize MatSize

	SavedDataMat
}

func NewAvgPool(poolSize, inputSize, stride [2]int) *AvgPool {
	return &AvgPool{
		poolSize:   MatSize{poolSize[0], poolSize[1]},
		stride:     Stride{inputSize[0], inputSize[1]},
		outputSize: MatSize{inputSize[0] / poolSize[0], inputSize[1] / poolSize[1]},
	}
}

func (layer *AvgPool) Forward(input *[]mat.Dense) *[]mat.Dense {
	return &layer.lastOutput
}
