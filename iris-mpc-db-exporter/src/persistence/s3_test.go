package persistence

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/stretchr/testify/require"
)

type failingHTTPClient struct{ err error }

func (c failingHTTPClient) Do(*http.Request) (*http.Response, error) { return nil, c.err }

type fakeS3WriterClient struct {
	createErr       error
	uploadErr       error
	completeErr     error
	abortErr        error
	createCalls     int
	uploadCalls     int
	completeCalls   int
	abortCalls      int
	putCalls        int
	abortContextErr error
	abortDeadline   time.Time
}

func (c *fakeS3WriterClient) CreateMultipartUpload(_ context.Context, input *s3.CreateMultipartUploadInput, _ ...func(*s3.Options)) (*s3.CreateMultipartUploadOutput, error) {
	c.createCalls++
	if c.createErr != nil {
		return nil, c.createErr
	}
	return &s3.CreateMultipartUploadOutput{Bucket: input.Bucket, Key: input.Key, UploadId: aws.String("upload-id")}, nil
}

func (c *fakeS3WriterClient) UploadPart(_ context.Context, _ *s3.UploadPartInput, _ ...func(*s3.Options)) (*s3.UploadPartOutput, error) {
	c.uploadCalls++
	if c.uploadErr != nil {
		return nil, c.uploadErr
	}
	return &s3.UploadPartOutput{ETag: aws.String("etag")}, nil
}

func (c *fakeS3WriterClient) CompleteMultipartUpload(_ context.Context, _ *s3.CompleteMultipartUploadInput, _ ...func(*s3.Options)) (*s3.CompleteMultipartUploadOutput, error) {
	c.completeCalls++
	return &s3.CompleteMultipartUploadOutput{}, c.completeErr
}

func (c *fakeS3WriterClient) AbortMultipartUpload(ctx context.Context, _ *s3.AbortMultipartUploadInput, _ ...func(*s3.Options)) (*s3.AbortMultipartUploadOutput, error) {
	c.abortCalls++
	c.abortContextErr = ctx.Err()
	c.abortDeadline, _ = ctx.Deadline()
	return &s3.AbortMultipartUploadOutput{}, c.abortErr
}

func (c *fakeS3WriterClient) PutObject(_ context.Context, _ *s3.PutObjectInput, _ ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	c.putCalls++
	return &s3.PutObjectOutput{}, nil
}

func streamInput(items ...[]byte) <-chan []byte {
	input := make(chan []byte, len(items))
	for _, item := range items {
		input <- item
	}
	close(input)
	return input
}

func terminalStatus(err error) <-chan error {
	status := make(chan error, 1)
	status <- err
	close(status)
	return status
}

func TestS3PersistStreamCompletesOnlyAfterProducerSuccess(t *testing.T) {
	client := &fakeS3WriterClient{}
	writer := &S3Writer{Client: client, Bucket: "bucket", MaxItemsPerPartUpload: 3}

	require.NoError(t, writer.PersistStream(context.Background(), "chunk", streamInput([]byte("iris-1"), []byte("iris-2")), terminalStatus(nil)))
	require.Equal(t, 1, client.createCalls)
	require.Equal(t, 1, client.uploadCalls)
	require.Equal(t, 1, client.completeCalls)
	require.Zero(t, client.abortCalls)
}

func TestS3PersistStreamAbortsOnProducerFailure(t *testing.T) {
	wantErr := errors.New("conversion failed")
	client := &fakeS3WriterClient{}
	writer := &S3Writer{Client: client, Bucket: "bucket", MaxItemsPerPartUpload: 1}

	err := writer.PersistStream(context.Background(), "chunk", streamInput([]byte("partial")), terminalStatus(wantErr))
	require.ErrorIs(t, err, wantErr)
	require.Equal(t, 1, client.uploadCalls)
	require.Zero(t, client.completeCalls)
	require.Equal(t, 1, client.abortCalls)
}

func TestS3PersistStreamRejectsMissingProducerStatus(t *testing.T) {
	client := &fakeS3WriterClient{}
	writer := &S3Writer{Client: client, Bucket: "bucket", MaxItemsPerPartUpload: 1}
	missingStatus := make(chan error)
	close(missingStatus)

	err := writer.PersistStream(context.Background(), "chunk", streamInput([]byte("partial")), missingStatus)
	require.ErrorIs(t, err, errMissingProducerStatus)
	require.Zero(t, client.completeCalls)
	require.Equal(t, 1, client.abortCalls)
}

func TestS3PersistStreamJoinsPrimaryAndAbortErrors(t *testing.T) {
	producerErr := errors.New("database stream failed")
	abortErr := errors.New("abort failed")
	client := &fakeS3WriterClient{abortErr: abortErr}
	writer := &S3Writer{Client: client, Bucket: "bucket", MaxItemsPerPartUpload: 1}

	err := writer.PersistStream(context.Background(), "chunk", streamInput([]byte("partial")), terminalStatus(producerErr))
	require.ErrorIs(t, err, producerErr)
	require.ErrorIs(t, err, abortErr)
	require.Equal(t, 1, client.abortCalls)
}

func TestS3PersistStreamAbortsWithBoundedUncancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := &fakeS3WriterClient{}
	writer := &S3Writer{Client: client, Bucket: "bucket", MaxItemsPerPartUpload: 1}
	input := make(chan []byte)

	err := writer.PersistStream(ctx, "chunk", input, terminalStatus(nil))
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, 1, client.abortCalls)
	require.NoError(t, client.abortContextErr)
	require.WithinDuration(t, time.Now().Add(10*time.Second), client.abortDeadline, time.Second)
}

func TestS3PersistStreamAbortsUploadAndCompleteFailures(t *testing.T) {
	for _, test := range []struct {
		name   string
		client *fakeS3WriterClient
	}{
		{name: "upload", client: &fakeS3WriterClient{uploadErr: errors.New("upload failed")}},
		{name: "complete", client: &fakeS3WriterClient{completeErr: errors.New("complete failed")}},
	} {
		t.Run(test.name, func(t *testing.T) {
			writer := &S3Writer{Client: test.client, Bucket: "bucket", MaxItemsPerPartUpload: 1}
			require.Error(t, writer.PersistStream(context.Background(), "chunk", streamInput([]byte("iris")), terminalStatus(nil)))
			require.Equal(t, 1, test.client.abortCalls)
		})
	}
}

func TestS3PersistenceErrors(t *testing.T) {
	wantErr := errors.New("storage unavailable")
	writer := &S3Writer{
		Client: s3.NewFromConfig(aws.Config{
			Region:           "eu-north-1",
			Credentials:      credentials.NewStaticCredentialsProvider("test", "test", ""),
			HTTPClient:       failingHTTPClient{wantErr},
			RetryMaxAttempts: 1,
		}),
		Bucket:                "test-bucket",
		MaxItemsPerPartUpload: 1,
	}
	require.ErrorIs(t, writer.Persist("timestamps/test", nil), wantErr)
	require.ErrorIs(t, writer.PersistStream(context.Background(), "chunk", streamInput(), terminalStatus(nil)), wantErr)
}
