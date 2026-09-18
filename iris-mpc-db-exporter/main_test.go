package main

import (
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRootCommandFailureExitsNonzero(t *testing.T) {
	if os.Getenv("EXPORTER_TEST_ROOT_FAILURE") == "1" {
		os.Args = []string{"app", "unknown-command"}
		main()
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRootCommandFailureExitsNonzero$")
	cmd.Env = append(os.Environ(), "EXPORTER_TEST_ROOT_FAILURE=1", "SLEEP_BEFORE_SHUTDOWN_SECONDS=0", "DD_TRACE_ENABLED=false")
	output, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	require.ErrorAs(t, err, &exitErr, string(output))
	require.Equal(t, 1, exitErr.ExitCode(), string(output))
	require.Contains(t, string(output), "unknown command")
}
