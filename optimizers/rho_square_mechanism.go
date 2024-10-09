package optimizers

import (
	"math"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/layers"
)

type rhoSquareMechanism struct {
	rho           float64
	rhoComplement float64
	eps           float64
	rootFunc      func(i, j int, v float64) float64

	rhoCorrection correctionMechanism
	denseSquared  map[int]*denseMomentum  // key: idx of dense layer in passed architecture
	convSquared   map[int]*filterMomentum // key: idx of conv layer in passed architecture
}

func newRhoSquareMechanism(rho, eps float64) rhoSquareMechanism {
	rootFunc_ := func(i, j int, v float64) float64 {
		if v == 0.0 {
			return math.Sqrt(v) + eps
		}
		return math.Sqrt(v)
	}
	return rhoSquareMechanism{
		rho:           rho,
		rhoComplement: 1.0 - rho,
		rootFunc:      rootFunc_,
		eps:           eps,
		rhoCorrection: correctionMechanism{
			decay:  rho,
			decayT: rho,
		},
	}
}

func (r *rhoSquareMechanism) initRhoMechanizm(
	convLayers *[]conv.ConvLayer,
	denseLayers *[]layers.Layer,
) {
	r.denseSquared = initDenseVelocities(denseLayers)
	r.convSquared = initConvVelocities(convLayers)
}

func (r *rhoSquareMechanism) rhoUpdateDense(idx int, dw2 *mat.Dense, db2 *mat.VecDense) {
	r.denseSquared[idx].updateWeight(dw2, &r.rho, &r.rhoComplement)
	r.denseSquared[idx].updateBias(db2, &r.rho, &r.rhoComplement)
}

func (r *rhoSquareMechanism) rhoUpdateConv(idx int, dw2 *[]mat.Dense, db2 *[]float64) {
	r.convSquared[idx].updateFilter(dw2, &r.rho, &r.rhoComplement)
	r.convSquared[idx].updateBias(db2, &r.rho, &r.rhoComplement)
}

func (m *rhoSquareMechanism) getRhoSquareDenseWeights(idx int) *mat.Dense {
	return &m.denseSquared[idx].weightsVelocities
}

func (m *rhoSquareMechanism) getRhoSquareDenseBias(idx int) *mat.VecDense {
	return &m.denseSquared[idx].biasesVelocities
}

func (m *rhoSquareMechanism) getRhoSquareConvWeigts(idx int) *[]mat.Dense {
	return &m.convSquared[idx].channelsVelocities
}

func (m *rhoSquareMechanism) getRhoSquareConvBias(idx int) *[]float64 {
	return &m.convSquared[idx].biasesVelocities
}

func (r *rhoSquareMechanism) zeroRhoActivate(v *float64) float64 {
	if *v > 0.0 {
		return 1.0
	} else if *v < 0.0 {
		return -1.0
	}
	return 0.0
}

func (r *rhoSquareMechanism) zeroRhoActivateDense(grads *mat.Dense) *mat.Dense {
	var retVal mat.Dense
	retVal.Apply(func(i, j int, v float64) float64 {
		return r.zeroRhoActivate(&v)
	}, grads)
	return &retVal
}

func (r *rhoSquareMechanism) zeroRhoActivateVec(grads *mat.VecDense) *mat.VecDense {
	retVal := mat.NewVecDense(grads.Len(), nil)
	for i, v := range grads.RawVector().Data {
		retVal.SetVec(i, r.zeroRhoActivate(&v))
	}
	return retVal
}

func (r *rhoSquareMechanism) zeroRhoActivateFloatSlice(grads *[]float64) *[]float64 {
	n := len(*grads)
	retVal := make([]float64, n)
	for i, v := range *grads {
		retVal[i] = r.zeroRhoActivate(&v)
	}
	return &retVal
}

func (r *rhoSquareMechanism) zeroRhoActivateDenseSlice(grads *[]mat.Dense) *[]mat.Dense {
	retVal := make([]mat.Dense, len(*grads))
	for i, gradDense := range *grads {
		retVal[i] = *r.zeroRhoActivateDense(&gradDense)
	}
	return &retVal
}

// TODO Preallocate space for squared grads instead of allocating every r step
func (r *rhoSquareMechanism) squareDenseLayerGrads(
	denseGrads *mat.Dense,
	biasGrads *mat.VecDense,
) (mat.Dense, mat.VecDense) {
	var denseGradsRet mat.Dense
	denseGradsRet.MulElem(
		denseGrads,
		denseGrads,
	)
	var vecBiasGradsRet mat.VecDense
	vecBiasGradsRet.MulElemVec(
		biasGrads,
		biasGrads,
	)
	return denseGradsRet, vecBiasGradsRet
}

// TODO Preallocate space for squared grads instead of allocating every r step
func (r *rhoSquareMechanism) squareConvLayerGrads(
	convGrads *[]mat.Dense,
	biasGrads *[]float64,
) ([]mat.Dense, []float64) {
	convGradsRet := make([]mat.Dense, len(*convGrads))
	convBiasRet := make([]float64, len(*biasGrads))
	for i, convGrad := range *convGrads {
		convGradsRet[i].MulElem(&convGrad, &convGrad)
	}
	for i, biasGrad := range *biasGrads {
		convBiasRet[i] = biasGrad * biasGrad
	}
	return convGradsRet, convBiasRet
}

func (r *rhoSquareMechanism) gradsScaleSquaredDense(dw, dw2 *mat.Dense) mat.Dense {
	var rootedS mat.Dense
	rootedS.Apply(r.rootFunc, dw2)
	var retVal mat.Dense
	retVal.DivElem(dw, &rootedS)
	return retVal
}

func (r *rhoSquareMechanism) gradsScaleSquaredVec(dw, dw2 *mat.VecDense) mat.VecDense {
	retVal := mat.NewVecDense(dw.Len(), nil)
	for i := range dw.Len() {
		toSquare := dw2.AtVec(i)
		if toSquare == 0.0 {
			toSquare = r.eps
		}
		v := dw.AtVec(i) / math.Sqrt(toSquare)
		retVal.SetVec(i, v)
	}
	return *retVal
}

func (r *rhoSquareMechanism) gradsScaleSquaredConv(dw, dw2 *[]mat.Dense) []mat.Dense {
	retVal := make([]mat.Dense, len(*dw))
	for i := range retVal {
		retVal[i] = r.gradsScaleSquaredDense(&(*dw)[i], &(*dw2)[i])
	}
	return retVal
}

func (r *rhoSquareMechanism) gradsScaleSquaredFloatSlice(dw, dw2 *[]float64) []float64 {
	retVal := make([]float64, len(*dw))
	for i := range retVal {
		toSquare := (*dw2)[i]
		if toSquare == 0.0 {
			toSquare += r.eps
		}
		retVal[i] = (*dw)[i] / math.Sqrt(toSquare)
	}
	return retVal
}
