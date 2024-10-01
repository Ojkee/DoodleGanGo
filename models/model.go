package models

type Model interface {
	Train()
	Test()

	Save(filePath string) error
	Load(filePath string) error
}
