package optimizers

import (
	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type SDG struct {
	learningRate       float64
	lastConvOutputSize conv.MatSize
}

func NewSDG(learningRate float64, lastConvOutputSize [2]int) SDG {
	if learningRate <= 0 {
		panic("NewSDG fail:\n\tLearning Rate can't be less or equal 0")
	}
	return SDG{
		learningRate:       learningRate,
		lastConvOutputSize: *conv.NewMatSize(lastConvOutputSize[0], lastConvOutputSize[1]),
	}
}

func (opt *SDG) BackwardDenseLayers(denses *[]layers.DenseLayer, loss *mat.VecDense) *mat.VecDense {
	return nil
}

func (opt *SDG) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
	gradsMat := vecToMat(
		denseGrads,
		opt.lastConvOutputSize.Height(),
		opt.lastConvOutputSize.Width(),
	)
	for i := len(*convs2D) - 1; i >= 0; i-- {
		gradsMat = *(*convs2D)[i].Backward(&gradsMat)
	}
	for i := range *convs2D {
		(*convs2D)[i].ApplyGrads(&opt.learningRate)
	}
}

func vecToMat(source *mat.VecDense, height, width int) []mat.Dense {
	channelPixels := height * width
	numChannels := len(source.RawVector().Data) / channelPixels
	retVal := make([]mat.Dense, numChannels)
	sourceData := source.RawVector().Data
	for i := range numChannels {
		retVal[i] = *mat.NewDense(
			height,
			width,
			sourceData[i*channelPixels:(i+1)*channelPixels],
		)
	}
	return retVal
}
