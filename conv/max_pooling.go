package conv

import (
	"gonum.org/v1/gonum/mat"
)

type MaxPool struct {
	ConvType
	poolSize    MatSize
	stride      Stride
	savedMaxPos [][][]mPos // C x H x W

	SavedGrads
}

type mPos struct {
	y int
	x int
}

func NewMaxPool(poolSize, inputSize, stride [2]int, numChannels int) MaxPool {
	outputSize_ := MatSize{
		height: (inputSize[0]-poolSize[0])/stride[0] + 1,
		width:  (inputSize[1]-poolSize[1])/stride[1] + 1,
	}
	savedMaxPos_ := make([][][]mPos, numChannels)
	for i := range savedMaxPos_ {
		savedMaxPos_[i] = make([][]mPos, outputSize_.height)
		for j := range savedMaxPos_[i] {
			savedMaxPos_[i][j] = make([]mPos, outputSize_.width)
		}
	}
	return MaxPool{
		ConvType: ConvType{
			inputSize:  MatSize{inputSize[0], inputSize[1]},
			outputSize: outputSize_,
		},
		poolSize:    MatSize{poolSize[0], poolSize[1]},
		stride:      Stride{stride[0], stride[1]},
		savedMaxPos: savedMaxPos_,
	}
}

func (layer *MaxPool) Forward(input *[]mat.Dense) *[]mat.Dense {
	layer.lastInput = *input
	result := make([]mat.Dense, len(*input))
	for channelIdx := range *input {
		currentPool := make([]float64, layer.outputSize.height*layer.outputSize.width)
		for i := range layer.outputSize.height {
			for j := range layer.outputSize.width {
				area := (*input)[channelIdx].Slice(
					i*layer.stride.vertical,
					layer.poolSize.height+i*layer.stride.vertical,
					j*layer.stride.horizontal,
					layer.poolSize.width+j*layer.stride.horizontal,
				)
				currMax := area.At(0, 0) - 1
				for ai := range layer.poolSize.height {
					for aj := range layer.poolSize.width {
						if area.At(ai, aj) > currMax {
							currMax = area.At(ai, aj)
							layer.savedMaxPos[channelIdx][i][j].y = ai + i*layer.poolSize.height
							layer.savedMaxPos[channelIdx][i][j].x = aj + j*layer.poolSize.width
						}
					}
				}
				currentPool[i*layer.outputSize.width+j] = currMax
			}
		}
		result[channelIdx] = *mat.NewDense(layer.outputSize.height, layer.outputSize.width, currentPool)
	}
	layer.lastOutput = result
	return &result
}

func buildMaxGradMat(grads *mat.Dense, positions *[][]mPos, retSize *MatSize) mat.Dense {
	n, m := grads.Dims()
	retData := make([]float64, retSize.FlatDim())
	for i := range n {
		for j := range m {
			y := (*positions)[i][j].y
			x := (*positions)[i][j].x
			retData[y*retSize.width+x] = grads.At(i, j)
		}
	}
	return *mat.NewDense(retSize.height, retSize.width, retData)
}

func (layer *MaxPool) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	layer.lastInGrads = *inGrads
	layer.lastOutGrads = make([]mat.Dense, len(*inGrads))
	for channelIdx := range *inGrads {
		layer.lastOutGrads[channelIdx] = buildMaxGradMat(
			&(*inGrads)[channelIdx],
			&layer.savedMaxPos[channelIdx],
			&layer.inputSize,
		)
	}

	return &layer.lastOutGrads
}

func (layer *MaxPool) DeflatOutGrads() *[]mat.Dense {
	return &layer.lastOutGrads
}

func (layer *MaxPool) DeflatOutput() *[]mat.Dense {
	return GetDeflatOutput(&layer.ConvType)
}

func (layer *MaxPool) FlatOutput() *[]float64 {
	return GetFlatOutput(&layer.ConvType)
}
