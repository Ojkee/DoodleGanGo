package optimizers

import (
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
)

type SGD struct {
	learningRate       float64
	momentum           float64
	momentumComplement float64

	lastConvOutputSize conv.MatSize

	convVelocities  map[int]*filterMomentum // key: idx of conv layer in passed architecture
	denseVelocities map[int]*denseMomentum  // key: idx of dense layer in passed architecture
}

func NewSGD(learningRate, momentum float64) SGD {
	if learningRate <= 0 {
		panic("NewSGD fail:\n\tLearning Rate can't be less or equal 0")
	}
	if momentum < 0 || momentum > 1 {
		panic("NewSGD fail:\n\tMomentum must be in range [0, 1]")
	}
	return SGD{
		learningRate:       learningRate,
		momentum:           momentum,
		momentumComplement: 1.0 - momentum,
	}
}

func (opt *SGD) PreTrainInit(
	lastConvOutputSize [2]int,
	convLayers *[]conv.ConvLayer,
	denseLayers *[]layers.Layer,
) {
	opt.lastConvOutputSize = *conv.NewMatSize(
		lastConvOutputSize[0],
		lastConvOutputSize[1],
	)
	if opt.momentum != 0.0 {
		opt.denseVelocities = initDenseVelocities(denseLayers)
		opt.convVelocities = initConvVelocities(convLayers)
	}
}

func (opt *SGD) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
	grads := *loss
	for i, denseLayer := range slices.Backward(*denses) {
		grads = *denseLayer.Backward(&grads)
		if trainableLayer, ok := denseLayer.(layers.LayerTrainable); ok {
			if opt.momentum == 0.0 {
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					trainableLayer.GetOutWeightsGrads(),
					trainableLayer.GetOutBiasGrads(),
				)
			} else {
				// Non Nesterov
				weightGrads := trainableLayer.GetOutWeightsGrads()
				biasGrads := trainableLayer.GetOutBiasGrads()
				opt.denseVelocities[i].updateWeight(weightGrads, &opt.momentum, &opt.momentumComplement)
				opt.denseVelocities[i].updateBias(biasGrads, &opt.momentum, &opt.momentumComplement)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					&opt.denseVelocities[i].weightsVelocities,
					&opt.denseVelocities[i].biasesVelocities,
				)
			}
		}
	}
	return &grads
}

func (opt *SGD) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
	gradsMat := functools.VecToMatSlice(
		denseGrads,
		opt.lastConvOutputSize.Height(),
		opt.lastConvOutputSize.Width(),
	)
	for i := len(*convs2D) - 1; i >= 0; i-- {
		gradsMat = *(*convs2D)[i].Backward(&gradsMat)
		if trainableLayer, ok := (*convs2D)[i].(conv.ConvLayerTrainable); ok {
			if opt.momentum == 0.0 {
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					trainableLayer.GetFilterGrads(),
					trainableLayer.GetBiasGrads(),
				)
			} else {
				filterGrads := trainableLayer.GetFilterGrads()
				biasGrads := trainableLayer.GetBiasGrads()
				opt.convVelocities[i].updateFilter(filterGrads, &opt.momentum, &opt.momentumComplement)
				opt.convVelocities[i].updateBias(biasGrads, &opt.momentum, &opt.momentumComplement)
				trainableLayer.ApplyGrads(
					&opt.learningRate,
					&opt.convVelocities[i].channelsVelocities,
					&opt.convVelocities[i].biasesVelocities,
				)
			}
		}
	}
}
