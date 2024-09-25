package optimizers

import (
	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type SGD struct {
	learningRate       float64
	lastConvOutputSize conv.MatSize
}

func NewSGD(learningRate float64, lastConvOutputSize [2]int) SGD {
	if learningRate <= 0 {
		panic("NewSGD fail:\n\tLearning Rate can't be less or equal 0")
	}
	return SGD{
		learningRate:       learningRate,
		lastConvOutputSize: *conv.NewMatSize(lastConvOutputSize[0], lastConvOutputSize[1]),
	}
}

func (opt *SGD) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
	grads := *loss
	for i := len(*denses) - 1; i >= 0; i-- {
		grads = *(*denses)[i].Backward(&grads)
		// (*denses)[i].ApplyGrads(&opt.learningRate , (*denses)[i]., dBiasGrad *mat.VecDense)
		(*denses)[i].ApplyGrads(
			&opt.learningRate,
			(*denses)[i].GetOutWeightsGrads(),
			(*denses)[i].GetOutBiasGrads(),
		)
	}
	return &grads
}

func (opt *SGD) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
	gradsMat := vecToMat(
		denseGrads,
		opt.lastConvOutputSize.Height(),
		opt.lastConvOutputSize.Width(),
	)
	for i := len(*convs2D) - 1; i >= 0; i-- {
		gradsMat = *(*convs2D)[i].Backward(&gradsMat)
		biasGrads := (*convs2D)[i].GetBiasGrads()
		(*convs2D)[i].ApplyGrads(&opt.learningRate, &gradsMat, biasGrads)
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
