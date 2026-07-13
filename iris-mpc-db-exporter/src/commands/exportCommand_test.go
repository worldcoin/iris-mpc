package commands

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
)

func TestValidateRerandExportRequest(t *testing.T) {
	assert.NoError(t, validateRerandExportRequest(
		IncrementalExport,
		"",
		iris.RerandStatus{},
	))
	assert.Error(t, validateRerandExportRequest(
		CompleteExport,
		"",
		iris.RerandStatus{Initialized: true},
	))
	assert.Error(t, validateRerandExportRequest(
		IncrementalExport,
		"",
		iris.RerandStatus{HasPositiveState: true},
	))
	assert.NoError(t, validateRerandExportRequest(
		CompleteExport,
		"expected-store",
		iris.RerandStatus{Initialized: true, HasPositiveState: true},
	))
	assert.Error(t, validateRerandExportRequest(
		IncrementalExport,
		"expected-store",
		iris.RerandStatus{},
	))
}

func TestRequiresLegacyExportRerandLock(t *testing.T) {
	tests := []struct {
		name    string
		mode    string
		storeID string
		want    bool
	}{
		{name: "legacy complete", mode: CompleteExport, want: true},
		{name: "incremental", mode: IncrementalExport, want: true},
		{name: "invalid incremental with store ID is still fenced", mode: IncrementalExport, storeID: "store", want: true},
		{name: "safe content-addressed complete", mode: CompleteExport, storeID: "store", want: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, requiresLegacyExportRerandLock(test.mode, test.storeID))
		})
	}
}
