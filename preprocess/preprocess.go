package preprocess

import (
	"errors"
	"os"

	"github.com/sbinet/npyio"
)

func GetSplitData(
	fileName string,
	numberOfSamples, stride int,
	trainRatio float32,
) ([][]uint8, [][]uint8, error) {
	rawData, err := GetRawData(fileName)
	if err != nil {
		return nil, nil, err
	}
	if numberOfSamples < 0 {
		return nil, nil, errors.New("Dataset size can't be negative")
	}
	numberOfSamples = min(numberOfSamples, len(rawData)/stride)
	reshapedData := Reshape(rawData[:numberOfSamples*stride], stride)
	trainSet, testSet := Split(reshapedData, trainRatio)
	return trainSet, testSet, nil
}

func GetRawData(fileName string) ([]uint8, error) {
	dataFile, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer dataFile.Close()

	npyReader, err := npyio.NewReader(dataFile)
	if err != nil {
		return nil, err
	}

	var raw []uint8
	err = npyReader.Read(&raw)
	if err != nil {
		return nil, err
	}
	return raw, nil
}

func Reshape(source []uint8, stride int) [][]uint8 {
	var result [][]uint8
	for i := 0; i+stride <= len(source); i += stride {
		start := i * stride
		end := start + stride
		result = append(result, source[start:end])
	}
	return result
}

func Split(source [][]uint8, trainRatio float32) ([][]uint8, [][]uint8) {
	trainSize := int(float32(len(source)) * trainRatio)
	return source[:trainSize], source[trainSize:]
}
