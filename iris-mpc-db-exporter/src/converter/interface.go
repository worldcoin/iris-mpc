package converter

import "github.com/worldcoin/iris-mpc-db-exporter/src/iris"

type Converter interface {
	GetExtension() string
	Convert(data []iris.StoredIris) ([]byte, error)
	ConvertSingle(data iris.StoredIris) ([]byte, error)
}
