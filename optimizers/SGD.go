package optimizers

import (
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type filterVelocity struct {
	channelsVelocities []mat.Dense
	biasesVelocities   []float64
}

func (vel *filterVelocity) updateFilter(
	newGrad *[]mat.Dense,
	momentum, momentumComplement *float64,
) {
	for i, nGrad := range *newGrad {
		vel.channelsVelocities[i].Scale(*momentum, &vel.channelsVelocities[i])
		var newGradScaled mat.Dense
		newGradScaled.Scale(*momentumComplement, &nGrad)
		vel.channelsVelocities[i].Add(&vel.channelsVelocities[i], &newGradScaled)
	}
}

func (vel *filterVelocity) updateBias(
	newGrad *[]float64,
	momentum, momentumComplement *float64,
) {
	for i, bGrad := range *newGrad {
		vel.biasesVelocities[i] *= *momentum
		bGrad *= *momentumComplement
		vel.biasesVelocities[i] += bGrad
	}
}

type denseVelocity struct {
	weightsVelocities mat.Dense
	biasesVelocities  mat.VecDense
}

func (vel *denseVelocity) updateWeight(newGrad *mat.Dense, momentum, momentumComplement *float64) {
	vel.weightsVelocities.Scale(*momentum, &vel.weightsVelocities)
	var newGradScaled mat.Dense
	newGradScaled.Scale(*momentumComplement, newGrad)
	vel.weightsVelocities.Add(&vel.weightsVelocities, &newGradScaled)
}

func (vel *denseVelocity) updateBias(newGrad *mat.VecDense, momentum, momentumComplement *float64) {
	vel.biasesVelocities.ScaleVec(*momentum, &vel.biasesVelocities)
	vel.biasesVelocities.AddScaledVec(&vel.biasesVelocities, *momentumComplement, newGrad)
}

type SGD struct {
	learningRate       float64
	momentum           float64
	momentumComplement float64

	lastConvOutputSize conv.MatSize

	convVelocities  map[int]*filterVelocity // key: idx of conv layer in passed architecture
	denseVelocities map[int]*denseVelocity  // key: idx of dense layer in passed architecture
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
		opt.initConvVelocities(convLayers)
		opt.initDenseVelocities(denseLayers)
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
	gradsMat := vecToMat(
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

func initConvKernelsVelocities(convLayer *conv.Conv2D) *filterVelocity {
	n, m := convLayer.GetKernelSize()
	nChannels := convLayer.NumChannels()
	channelsVels := make([]mat.Dense, nChannels)
	for c := range nChannels {
		channelsVels[c] = *mat.NewDense(n, m, nil)
	}

	nFilters := convLayer.NumFilters()
	biasesVels := make([]float64, nFilters)
	for f := range nFilters {
		biasesVels[f] = 0.0
	}
	return &filterVelocity{
		channelsVelocities: channelsVels,
		biasesVelocities:   biasesVels,
	}
}

func (opt *SGD) initConvVelocities(convLayers *[]conv.ConvLayer) {
	opt.convVelocities = make(map[int]*filterVelocity)
	for i, convTypeLayer := range *convLayers {
		if convLayer, ok := convTypeLayer.(*conv.Conv2D); ok {
			opt.convVelocities[i] = initConvKernelsVelocities(convLayer)
		}
	}
}

func (opt *SGD) initDenseVelocities(denseLayers *[]layers.Layer) {
	opt.denseVelocities = make(map[int]*denseVelocity)
	for i, denseTypeLayer := range *denseLayers {
		if denseLayer, ok := denseTypeLayer.(*layers.DenseLayer); ok {
			n, m := denseLayer.WeightsSize()
			opt.denseVelocities[i] = &denseVelocity{
				weightsVelocities: *mat.NewDense(n, m, nil),
				biasesVelocities:  *mat.NewVecDense(n, nil),
			}
		}
	}
}
