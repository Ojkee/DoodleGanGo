package conv

import (
	"gonum.org/v1/gonum/mat"

	"DoodleGan/functools"
)

type AvgPool struct {
	ConvType
	poolSize MatSize
	stride   Stride
	matPool  mat.Dense

	SavedGrads
}

func NewAvgPool(poolSize, inputSize, stride [2]int) AvgPool {
	outputSize := [2]int{
		(inputSize[0]-poolSize[0])/stride[0] + 1,
		(inputSize[1]-poolSize[1])/stride[1] + 1,
	}
	return AvgPool{
		ConvType: ConvType{
			inputSize:  MatSize{inputSize[0], inputSize[1]},
			outputSize: MatSize{outputSize[0], outputSize[1]},
		},
		poolSize: MatSize{poolSize[0], poolSize[1]},
		stride:   Stride{stride[0], stride[1]},
		matPool:  PrepareToPool(&inputSize, &outputSize, &poolSize, &stride),
	}
}

func PrepareToPool(inputSize, outputSize, poolSize *[2]int, stride *[2]int) mat.Dense {
	inputFlatDim := inputSize[0] * inputSize[1]
	poolValues := make([]float64, outputSize[0]*outputSize[1]*inputFlatDim)
	rowOffset := 0
	poolVal := float64(1.0 / float64(poolSize[0]*poolSize[1]))
	for range outputSize[0] {
		for j := range outputSize[1] {
			for ki := range poolSize[0] {
				for kj := range poolSize[1] {
					c := rowOffset + j*stride[1] + kj + ki*inputSize[1]
					poolValues[c] = poolVal
				}
			}
			rowOffset += inputSize[0] * inputSize[1]
		}
		rowOffset += inputSize[1] * stride[0]
	}
	return *mat.NewDense(outputSize[0]*outputSize[1], inputFlatDim, poolValues)
}

func (layer *AvgPool) PrintFilter(precision int) {
	functools.PrintMat(&layer.matPool, precision)
}

func (layer *AvgPool) Forward(input *[]mat.Dense) *[]mat.Dense {
	inputFlatDim := layer.inputSize.height * layer.inputSize.width

	layer.lastInput = *input
	layer.lastOutput = make([]mat.Dense, len(*input))

	for i := range len(*input) {
		flatInput := *mat.NewVecDense(inputFlatDim, functools.FlattenMat(&(*input)[i]))
		var pm mat.Dense
		pm.Mul(&layer.matPool, &flatInput)
		layer.lastOutput[i] = pm
	}
	return &layer.lastOutput
}

func buildAvgGradMat(grads *mat.Dense, retSize, poolSize *MatSize) mat.Dense {
	retData := make([]float64, retSize.FlatDim())
	n, m := grads.Dims()
	scaler := float64(1.0 / (float64(poolSize.FlatDim())))
	for i := range n {
		for j := range m {
			cGrad := grads.At(i, j) * scaler
			for pi := range poolSize.height {
				for pj := range poolSize.width {
					offset := (i*poolSize.height+pi)*retSize.width + j*poolSize.width + pj
					retData[offset] = cGrad
				}
			}
		}
	}
	return *mat.NewDense(retSize.height, retSize.width, retData)
}

func (layer *AvgPool) Backward(inGrads *[]mat.Dense) *[]mat.Dense {
	layer.lastInGrads = *inGrads
	layer.lastOutGrads = make([]mat.Dense, len(*inGrads))
	for i, grad := range *inGrads {
		layer.lastOutGrads[i] = *mat.NewDense(
			layer.inputSize.height,
			layer.inputSize.width,
			nil,
		)
		avgPoolGrad := buildAvgGradMat(&grad, &layer.inputSize, &layer.poolSize)
		layer.lastOutGrads[i].Add(
			&layer.lastOutGrads[i],
			&avgPoolGrad,
		)
	}
	return &layer.lastOutGrads
}

func (layer *AvgPool) ApplyGrads(
	learningRate *float64,
	dWeightsGrads *[]mat.Dense,
	dBiasGrad *[]float64,
) {
	return
}

func (layer *AvgPool) DeflatOutGrads() *[]mat.Dense {
	return &layer.lastOutGrads
}

func (layer *AvgPool) GetBiasGrads() *[]float64 {
	return nil
}

func (layer *AvgPool) DeflatOutput() *[]mat.Dense {
	return GetDeflatOutput(&layer.ConvType)
}

func (layer *AvgPool) FlatOutput() *[]float64 {
	return GetFlatOutput(&layer.ConvType)
}
