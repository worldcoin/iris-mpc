package persistence

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/stretchr/testify/require"
)

type failingHTTPClient struct{ err error }

func (c failingHTTPClient) Do(*http.Request) (*http.Response, error) { return nil, c.err }

func TestS3PersistenceErrors(t *testing.T) {
	wantErr := errors.New("storage unavailable")
	writer := &S3Writer{
		Client: s3.NewFromConfig(aws.Config{
			Region:           "eu-north-1",
			Credentials:      credentials.NewStaticCredentialsProvider("test", "test", ""),
			HTTPClient:       failingHTTPClient{wantErr},
			RetryMaxAttempts: 1,
		}),
		Bucket: "test-bucket",
	}
	require.ErrorIs(t, writer.Persist("timestamps/test", nil), wantErr)
	input := make(chan []byte)
	close(input)
	require.ErrorIs(t, writer.PersistStream(context.Background(), "chunk", input), wantErr)
}
