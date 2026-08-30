package commands

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Format 1 keeps the historical 3-part marker name so existing importers are
// unaffected; newer formats append the version as a 4th part.
func TestTimestampMarkerName(t *testing.T) {
	assert.Equal(t, "1720000000_1000_954", timestampMarkerName(1720000000, 1000, 954, 1))
	assert.Equal(t, "1720000000_1000_954_2", timestampMarkerName(1720000000, 1000, 954, 2))
}
