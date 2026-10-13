package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

func TestExportFailureDoesNotPrintUsage(t *testing.T) {
	exportCmd := exportCommand()
	exportCmd.RunE = func(_ *cobra.Command, _ []string) error {
		return errors.New("upload failed: access denied")
	}

	rootCmd := &cobra.Command{Use: "app"}
	rootCmd.AddCommand(exportCmd)
	rootCmd.SetArgs([]string{"export-db"})
	var output bytes.Buffer
	rootCmd.SetOut(&output)
	rootCmd.SetErr(&output)

	err := rootCmd.Execute()
	require.ErrorContains(t, err, "upload failed: access denied")
	require.Contains(t, output.String(), "upload failed: access denied")
	require.NotContains(t, output.String(), "Usage:")
}

func TestExportFlagErrorPrintsUsage(t *testing.T) {
	exportCmd := exportCommand()
	exportCmd.RunE = func(_ *cobra.Command, _ []string) error {
		t.Fatal("export ran despite an invalid flag")
		return nil
	}

	rootCmd := &cobra.Command{Use: "app"}
	rootCmd.AddCommand(exportCmd)
	rootCmd.SetArgs([]string{"export-db", "--export-ouput=s3"})
	var output bytes.Buffer
	rootCmd.SetOut(&output)
	rootCmd.SetErr(&output)

	err := rootCmd.Execute()
	require.ErrorContains(t, err, "unknown flag: --export-ouput")
	require.Contains(t, output.String(), "Usage:")
	require.Contains(t, output.String(), "--export-output")
}

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
