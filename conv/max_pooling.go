package conv

import (
	"gonum.org/v1/gonum/mat"
)

type MaxPool struct {
	ConvType
	poolSize MatSize
	stride   Stride
}

func NewMaxPool(poolSize, inputSize, stride [2]int) *MaxPool {
	outputSize := [2]int{
		(inputSize[0]-poolSize[0])/stride[0] + 1,
		(inputSize[1]-poolSize[1])/stride[1] + 1,
	}
	return &MaxPool{
		ConvType: ConvType{
			inputSize:  MatSize{inputSize[0], inputSize[1]},
			outputSize: MatSize{outputSize[0], outputSize[1]},
		},
		poolSize: MatSize{poolSize[0], poolSize[1]},
		stride:   Stride{stride[0], stride[1]},
	}
}

func (layer *MaxPool) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	result := make([]mat.Dense, len(*input))
	for inputIndex := range len(*input) {
		currentPool := make([]float64, layer.outputSize.height*layer.outputSize.width)
		for i := range layer.outputSize.height {
			for j := range layer.outputSize.width {
				currentPool[i*layer.outputSize.width+j] = mat.Max(
					(*input)[inputIndex].Slice(
						i*layer.stride.vertical,
						layer.poolSize.height+i*layer.stride.vertical,
						j*layer.stride.horizontal,
						layer.poolSize.width+j*layer.stride.horizontal,
					))
			}
		}
		result[inputIndex] = *mat.NewDense(layer.outputSize.height, layer.outputSize.width, currentPool)
	}
	layer.lastOutput = result
	return &result
}

func (layer *MaxPool) DeflatOutput() *[]mat.Dense {
	return GetDeflatOutput(&layer.ConvType)
}

func (layer *MaxPool) FlatOutput() *[]float64 {
	return GetFlatOutput(&layer.ConvType)
}
