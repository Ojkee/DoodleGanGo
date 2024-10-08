package optimizers

import (
	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type momentumMechanism struct {
	momentum           float64
	momentumComplement float64

	velocityCorrection correctionMechanism
	convVelocities     map[int]*filterMomentum // key: idx of conv layer in passed architecture
	denseVelocities    map[int]*denseMomentum  // key: idx of dense layer in passed architecture
}

func newMomentumMechanism(momentum float64) momentumMechanism {
	return momentumMechanism{
		momentum:           momentum,
		momentumComplement: 1.0 - momentum,
		velocityCorrection: correctionMechanism{
			decay:  momentum,
			decayT: momentum,
		},
	}
}

func (m *momentumMechanism) initMomentumMechanizm(
	convLayers *[]conv.ConvLayer,
	denseLayers *[]layers.Layer,
) {
	m.denseVelocities = initDenseVelocities(denseLayers)
	m.convVelocities = initConvVelocities(convLayers)
}

func (m *momentumMechanism) momentumUpdateDense(idx int, dw *mat.Dense, db *mat.VecDense) {
	m.denseVelocities[idx].updateWeight(dw, &m.momentum, &m.momentumComplement)
	m.denseVelocities[idx].updateBias(db, &m.momentum, &m.momentumComplement)
}

func (m *momentumMechanism) getMomentumDenseWeights(idx int) *mat.Dense {
	return &m.denseVelocities[idx].weightsVelocities
}

func (m *momentumMechanism) getMomentumDenseBias(idx int) *mat.VecDense {
	return &m.denseVelocities[idx].biasesVelocities
}

func (m *momentumMechanism) momentumUpdateConv(idx int, dw *[]mat.Dense, db *[]float64) {
	m.convVelocities[idx].updateFilter(dw, &m.momentum, &m.momentumComplement)
	m.convVelocities[idx].updateBias(db, &m.momentum, &m.momentumComplement)
}

func (m *momentumMechanism) getMomentumConvWeigts(idx int) *[]mat.Dense {
	return &m.convVelocities[idx].channelsVelocities
}

func (m *momentumMechanism) getMomentumConvBias(idx int) *[]float64 {
	return &m.convVelocities[idx].biasesVelocities
}
