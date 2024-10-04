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

	denseSquared map[int]*denseMomentum  // key: idx of dense layer in passed architecture
	convSquared  map[int]*filterMomentum // key: idx of conv layer in passed architecture
}

func newRhoSquareMechanism(rho, eps float64) rhoSquareMechanism {
	rootFunc_ := func(i, j int, v float64) float64 {
		if v == 0.0 {
			v += eps
		}
		return math.Sqrt(v)
	}
	return rhoSquareMechanism{
		rho:           rho,
		rhoComplement: 1.0 - rho,
		rootFunc:      rootFunc_,
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
	var retVal mat.VecDense
	for i, v := range grads.RawVector().Data {
		retVal.SetVec(i, r.zeroRhoActivate(&v))
	}
	return &retVal
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

// TODO Preallocate space for squared grads instead of allocating every opt step
func (r *rhoSquareMechanism) squareDenseLayerGrads(
	denseGrads *mat.Dense,
	biasGrads *mat.VecDense,
) (mat.Dense, mat.VecDense) {
	var denseGradsRet mat.Dense
	denseGrads.MulElem(
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

// TODO Preallocate space for squared grads instead of allocating every opt step
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

func (opt *rhoSquareMechanism) gradsScaleSquaredDense(grads, gradsS *mat.Dense) mat.Dense {
	var rootedS mat.Dense
	rootedS.Apply(opt.rootFunc, gradsS)
	var retVal mat.Dense
	retVal.DivElem(grads, &rootedS)
	return retVal
}

func (opt *rhoSquareMechanism) gradsScaleSquaredVec(grads, gradsS *mat.VecDense) mat.VecDense {
	retVal := mat.NewVecDense(grads.Len(), nil)
	for i := range grads.Len() {
		toSquare := gradsS.AtVec(i)
		if toSquare == 0.0 {
			toSquare += opt.eps
		}
		retVal.SetVec(i, grads.AtVec(i)/math.Sqrt(toSquare))
	}
	return *retVal
}

func (opt *rhoSquareMechanism) gradsScaleSquaredConv(grads, gradsS *[]mat.Dense) []mat.Dense {
	retVal := make([]mat.Dense, len(*grads))
	for i := range retVal {
		retVal[i] = opt.gradsScaleSquaredDense(&(*grads)[i], &(*gradsS)[i])
	}
	return retVal
}

func (opt *rhoSquareMechanism) gradsScaleSquaredFloatSlice(grads, gradsS *[]float64) []float64 {
	retVal := make([]float64, len(*grads))
	for i := range retVal {
		toSquare := (*gradsS)[i]
		if toSquare == 0.0 {
			toSquare += opt.eps
		}
		retVal[i] = (*grads)[i] / math.Sqrt(toSquare)
	}
	return retVal
}
