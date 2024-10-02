package optimizers

import (
	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type filterMomentum struct {
	channelsVelocities []mat.Dense
	biasesVelocities   []float64
}

func (m *filterMomentum) updateFilter(
	newGrad *[]mat.Dense,
	momentum, momentumComplement *float64,
) {
	for i, nGrad := range *newGrad {
		m.channelsVelocities[i].Scale(*momentum, &m.channelsVelocities[i])
		var newGradScaled mat.Dense
		newGradScaled.Scale(*momentumComplement, &nGrad)
		m.channelsVelocities[i].Add(&m.channelsVelocities[i], &newGradScaled)
	}
}

func (m *filterMomentum) updateBias(
	newGrad *[]float64,
	momentum, momentumComplement *float64,
) {
	for i, bGrad := range *newGrad {
		m.biasesVelocities[i] *= *momentum
		bGrad *= *momentumComplement
		m.biasesVelocities[i] += bGrad
	}
}

type denseMomentum struct {
	weightsVelocities mat.Dense
	biasesVelocities  mat.VecDense
}

func (m *denseMomentum) updateWeight(newGrad *mat.Dense, momentum, momentumComplement *float64) {
	m.weightsVelocities.Scale(*momentum, &m.weightsVelocities)
	var newGradScaled mat.Dense
	newGradScaled.Scale(*momentumComplement, newGrad)
	m.weightsVelocities.Add(&m.weightsVelocities, &newGradScaled)
}

func (m *denseMomentum) updateBias(newGrad *mat.VecDense, momentum, momentumComplement *float64) {
	m.biasesVelocities.ScaleVec(*momentum, &m.biasesVelocities)
	m.biasesVelocities.AddScaledVec(&m.biasesVelocities, *momentumComplement, newGrad)
}

func initConvKernelsVelocities(convLayer *conv.Conv2D) *filterMomentum {
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
	return &filterMomentum{
		channelsVelocities: channelsVels,
		biasesVelocities:   biasesVels,
	}
}

func initConvVelocities(convLayers *[]conv.ConvLayer) map[int]*filterMomentum {
	retVal := make(map[int]*filterMomentum)
	for i, convTypeLayer := range *convLayers {
		if convLayer, ok := convTypeLayer.(*conv.Conv2D); ok {
			retVal[i] = initConvKernelsVelocities(convLayer)
		}
	}
	return retVal
}

func initDenseVelocities(denseLayers *[]layers.Layer) map[int]*denseMomentum {
	retVal := make(map[int]*denseMomentum)
	for i, denseTypeLayer := range *denseLayers {
		if denseLayer, ok := denseTypeLayer.(*layers.DenseLayer); ok {
			n, m := denseLayer.WeightsSize()
			retVal[i] = &denseMomentum{
				weightsVelocities: *mat.NewDense(n, m, nil),
				biasesVelocities:  *mat.NewVecDense(n, nil),
			}
		}
	}
	return retVal
}
