package optimizers

import "gonum.org/v1/gonum/mat"

type correctionType interface {
	correctedDense(dw *mat.Dense) mat.Dense
	correctedDenseSlice(dw *[]mat.Dense) []mat.Dense
	correctedVecDense(db *mat.VecDense) mat.VecDense
	correctedFloatSlice(db *[]float64) []float64
}

type correctionMechanism struct {
	decay  float64
	decayT float64
}

func (c *correctionMechanism) updateDecayT() {
	c.decayT *= c.decay
}

func (c *correctionMechanism) scaleFraction() float64 {
	return 1.0 / (1.0 - c.decayT)
}

func (c *correctionMechanism) correctedDense(dw *mat.Dense) mat.Dense {
	var retVal mat.Dense
	retVal.Scale(c.scaleFraction(), dw)
	c.updateDecayT()
	return retVal
}

func (c *correctionMechanism) correctedDenseSlice(dw *[]mat.Dense) []mat.Dense {
	retVal := make([]mat.Dense, len(*dw))
	cScaleFraction := c.scaleFraction()
	for i, dChannel := range *dw {
		retVal[i].Scale(cScaleFraction, &dChannel)
	}
	c.updateDecayT()
	return retVal
}

func (c correctionMechanism) correctedVecDense(db *mat.VecDense) mat.VecDense {
	var retVal mat.VecDense
	cScaleFraction := c.scaleFraction()
	retVal.ScaleVec(cScaleFraction, db)
	c.updateDecayT()
	return retVal
}

func (c correctionMechanism) correctedFloatSlice(db *[]float64) []float64 {
	retVal := make([]float64, len(*db))
	cScaleFraction := c.scaleFraction()
	for i, v := range *db {
		retVal[i] = v * cScaleFraction
	}
	return retVal
}
