package conv

import "gonum.org/v1/gonum/mat"

type ConvLayer interface {
	Forward(*[]mat.Dense) *[]mat.Dense
}

type ConvType struct {
	SavedDataMat
	inputSize  MatSize
	outputSize MatSize
}

type SavedDataMat struct {
	lastInput  []mat.Dense
	lastOutput []mat.Dense
}

type MatSize struct {
	height int
	width  int
}

type Stride struct {
	horizontal int
	vertical   int
}

type Padding struct {
	up    int
	right int
	down  int
	left  int
}

func GetFlatOutput(layer *ConvType) *[]float64 {
	channelSize := layer.outputSize.height * layer.outputSize.width
	result := make([]float64, channelSize*len(layer.lastOutput))
	for i := range layer.lastOutput {
		for j := range channelSize {
			result[i*channelSize+j] = layer.lastOutput[i].RawMatrix().Data[j]
		}
	}
	return &result
}

func GetDeflatOutput(layer *ConvType) *[]mat.Dense {
	result := make([]mat.Dense, 0)
	for i := range layer.lastOutput {
		reshaped := mat.NewDense(
			layer.outputSize.height,
			layer.outputSize.width,
			layer.lastOutput[i].RawMatrix().Data,
		)
		result = append(result, *reshaped)
	}
	return &result
}

func (size *MatSize) FlatDim() int {
	return size.height * size.width
}
