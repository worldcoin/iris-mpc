package converter

import "github.com/worldcoin/iris-mpc-db-exporter/src/iris"

type Converter interface {
	GetExtension() string
	// FormatVersion identifies the on-disk record layout (see the Format*
	// constants); it is appended to the snapshot marker name for versions > 1.
	FormatVersion() int
	Convert(data []iris.StoredIris) ([]byte, error)
	ConvertSingle(data iris.StoredIris) ([]byte, error)
}
