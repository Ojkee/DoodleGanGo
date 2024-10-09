package optimizers

import (
	"slices"

	"gonum.org/v1/gonum/mat"

	"DoodleGan/conv"
	"DoodleGan/functools"
	"DoodleGan/layers"
)

type Adam struct {
	learningRate float64

	momentumMechanism
	rhoSquareMechanism
	lastConvOutputSize conv.MatSize
}

func NewAdam(learningRate, momentum, rho, eps float64) Adam {
	checkValidLearningRate(&learningRate, "NewAdam")
	checkValidMomentum(&momentum, "NewAdam")
	checkValidRho(&rho, "NewAdam")
	checkValidEps(&eps, "NewAdam")
	return Adam{
		learningRate:       learningRate,
		momentumMechanism:  newMomentumMechanism(momentum),
		rhoSquareMechanism: newRhoSquareMechanism(rho, eps),
	}
}

func (opt *Adam) PreTrainInit(
	lastConvOutputSize [2]int,
	convLayers *[]conv.ConvLayer,
	denseLayers *[]layers.Layer,
) {
	opt.lastConvOutputSize = *conv.NewMatSize(
		lastConvOutputSize[0],
		lastConvOutputSize[1],
	)
	if opt.rho != 0.0 {
		opt.initRhoMechanizm(convLayers, denseLayers)
	}
	if opt.momentum != 0.0 {
		opt.initMomentumMechanizm(convLayers, denseLayers)
	}
}

func (opt *Adam) BackwardDenseLayers(denses *[]layers.Layer, loss *mat.VecDense) *mat.VecDense {
	grads := *loss
	for i, denseLayer := range slices.Backward(*denses) {
		grads = *denseLayer.Backward(&grads)
		if trainableLayer, ok := denseLayer.(layers.LayerTrainable); ok {
			if opt.rho == 0.0 && opt.momentum == 0.0 {
				dw := opt.zeroRhoActivateDense(trainableLayer.GetOutWeightsGrads())
				db := opt.zeroRhoActivateVec(trainableLayer.GetOutBiasGrads())
				trainableLayer.ApplyGrads(&opt.learningRate, dw, db)
			} else if opt.momentum == 0.0 {
				dw := trainableLayer.GetOutWeightsGrads()
				db := trainableLayer.GetOutBiasGrads()
				dw2, db2 := opt.squareDenseLayerGrads(dw, db)
				opt.rhoUpdateDense(i, &dw2, &db2)
				dwCorrected := opt.rhoCorrection.correctedDense(opt.getRhoSquareDenseWeights(i))
				dbCorrected := opt.rhoCorrection.correctedVecDense(opt.getRhoSquareDenseBias(i))
				dwScaled := opt.gradsScaleSquaredDense(dw, &dwCorrected)
				dbScaled := opt.gradsScaleSquaredVec(db, &dbCorrected)
				trainableLayer.ApplyGrads(&opt.learningRate, &dwScaled, &dbScaled)
			} else if opt.rho == 0.0 {
				dw := trainableLayer.GetOutWeightsGrads()
				db := trainableLayer.GetOutBiasGrads()
				opt.momentumUpdateDense(i, dw, db)
				dwCorrected := opt.velocityCorrection.correctedDense(opt.getMomentumDenseWeights(i))
				dbCorrected := opt.velocityCorrection.correctedVecDense(opt.getMomentumDenseBias(i))
				dwScaled := opt.gradsScaleSquaredDense(&dwCorrected, dw)
				dbScaled := opt.gradsScaleSquaredVec(&dbCorrected, db)
				trainableLayer.ApplyGrads(&opt.learningRate, &dwScaled, &dbScaled)
			} else {
				dw := trainableLayer.GetOutWeightsGrads()
				db := trainableLayer.GetOutBiasGrads()
				opt.momentumUpdateDense(i, dw, db)
				dw2, db2 := opt.squareDenseLayerGrads(dw, db)
				opt.rhoUpdateDense(i, &dw2, &db2)
				vwCorrected := opt.velocityCorrection.correctedDense(opt.getMomentumDenseWeights(i))
				vbCorrected := opt.velocityCorrection.correctedVecDense(opt.getMomentumDenseBias(i))
				swCorrected := opt.rhoCorrection.correctedDense(opt.getRhoSquareDenseWeights(i))
				sbCorrected := opt.rhoCorrection.correctedVecDense(opt.getRhoSquareDenseBias(i))
				dwScaled := opt.gradsScaleSquaredDense(&vwCorrected, &swCorrected)
				dbScaled := opt.gradsScaleSquaredVec(&vbCorrected, &sbCorrected)
				trainableLayer.ApplyGrads(&opt.learningRate, &dwScaled, &dbScaled)
			}
		}
	}
	return &grads
}

func (opt *Adam) BackwardConv2DLayers(convs2D *[]conv.ConvLayer, denseGrads *mat.VecDense) {
	gradsMat := functools.VecToMatSlice(
		denseGrads,
		opt.lastConvOutputSize.Height(),
		opt.lastConvOutputSize.Width(),
	)
	for i, convLayer := range slices.Backward(*convs2D) {
		gradsMat = *convLayer.Backward(&gradsMat)
		if trainableLayer, ok := convLayer.(conv.ConvLayerTrainable); ok {
			if opt.rho == 0.0 && opt.momentum == 0.0 {
				dw := opt.zeroRhoActivateDenseSlice(trainableLayer.GetFilterGrads())
				db := opt.zeroRhoActivateFloatSlice(trainableLayer.GetBiasGrads())
				trainableLayer.ApplyGrads(&opt.learningRate, dw, db)
			} else if opt.momentum == 0.0 {
				dw := trainableLayer.GetFilterGrads()
				db := trainableLayer.GetBiasGrads()
				dw2, db2 := opt.squareConvLayerGrads(dw, db)
				opt.rhoUpdateConv(i, &dw2, &db2)
				dwCorrected := opt.rhoCorrection.correctedDenseSlice(opt.getRhoSquareConvWeigts(i))
				dbCorrected := opt.rhoCorrection.correctedFloatSlice(opt.getRhoSquareConvBias(i))
				dwScaled := opt.gradsScaleSquaredConv(dw, &dwCorrected)
				dbScaled := opt.gradsScaleSquaredFloatSlice(db, &dbCorrected)
				trainableLayer.ApplyGrads(&opt.learningRate, &dwScaled, &dbScaled)
			} else if opt.rho == 0.0 {
				dw := trainableLayer.GetFilterGrads()
				db := trainableLayer.GetBiasGrads()
				opt.momentumUpdateConv(i, dw, db)
				dwCorrected := opt.velocityCorrection.correctedDenseSlice(opt.getMomentumConvWeigts(i))
				dbCorrected := opt.velocityCorrection.correctedFloatSlice(opt.getMomentumConvBias(i))
				dwScaled := opt.gradsScaleSquaredConv(&dwCorrected, dw)
				dbScaled := opt.gradsScaleSquaredFloatSlice(&dbCorrected, db)
				trainableLayer.ApplyGrads(&opt.learningRate, &dwScaled, &dbScaled)
			} else {
				dw := trainableLayer.GetFilterGrads()
				db := trainableLayer.GetBiasGrads()
				opt.momentumUpdateConv(i, dw, db)
				dw2, db2 := opt.squareConvLayerGrads(dw, db)
				opt.rhoUpdateConv(i, &dw2, &db2)
				vwCorrected := opt.velocityCorrection.correctedDenseSlice(opt.getMomentumConvWeigts(i))
				vbCorrected := opt.velocityCorrection.correctedFloatSlice(opt.getMomentumConvBias(i))
				swCorrected := opt.rhoCorrection.correctedDenseSlice(opt.getRhoSquareConvWeigts(i))
				sbCorrected := opt.rhoCorrection.correctedFloatSlice(opt.getRhoSquareConvBias(i))
				dwScaled := opt.gradsScaleSquaredConv(&vwCorrected, &swCorrected)
				dbScaled := opt.gradsScaleSquaredFloatSlice(&vbCorrected, &sbCorrected)
				trainableLayer.ApplyGrads(&opt.learningRate, &dwScaled, &dbScaled)
			}
		}
	}
}
