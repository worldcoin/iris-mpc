package persistence

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"go.uber.org/zap"

	awsConfig "github.com/worldcoin/iris-mpc-db-exporter/src/aws"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
)

type s3WriterClient interface {
	CreateMultipartUpload(context.Context, *s3.CreateMultipartUploadInput, ...func(*s3.Options)) (*s3.CreateMultipartUploadOutput, error)
	UploadPart(context.Context, *s3.UploadPartInput, ...func(*s3.Options)) (*s3.UploadPartOutput, error)
	CompleteMultipartUpload(context.Context, *s3.CompleteMultipartUploadInput, ...func(*s3.Options)) (*s3.CompleteMultipartUploadOutput, error)
	AbortMultipartUpload(context.Context, *s3.AbortMultipartUploadInput, ...func(*s3.Options)) (*s3.AbortMultipartUploadOutput, error)
	PutObject(context.Context, *s3.PutObjectInput, ...func(*s3.Options)) (*s3.PutObjectOutput, error)
}

type S3Writer struct {
	Client                s3WriterClient
	Bucket                string
	MaxItemsPerPartUpload int
}

func (s *S3Writer) PersistStream(ctx context.Context, path string, inputChannel <-chan []byte, producerStatus <-chan error) (resultErr error) {
	if s.MaxItemsPerPartUpload <= 0 {
		return errors.New("max items per part upload must be positive")
	}

	input := &s3.CreateMultipartUploadInput{
		Bucket: &s.Bucket,
		Key:    &path,
	}
	resp, err := s.Client.CreateMultipartUpload(ctx, input)
	if err != nil {
		return fmt.Errorf("create multipart upload for %s: %w", path, err)
	}
	completed := false
	defer func() {
		if completed {
			return
		}

		abortCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		_, abortErr := s.Client.AbortMultipartUpload(abortCtx, &s3.AbortMultipartUploadInput{
			Bucket:   resp.Bucket,
			Key:      resp.Key,
			UploadId: resp.UploadId,
		})
		if abortErr != nil {
			resultErr = errors.Join(resultErr, fmt.Errorf("abort multipart upload for %s: %w", path, abortErr))
		}
	}()

	o11y.S(ctx).Infof("Created multipart upload with ID %s", *resp.UploadId)

	var completedParts []types.CompletedPart
	var outputBuffer []byte

	partNumber := int32(1)
	itemIdx := 0

	for {
		var item []byte
		var ok bool
		select {
		case <-ctx.Done():
			return ctx.Err()
		case item, ok = <-inputChannel:
		}
		if ok {
			outputBuffer = append(outputBuffer, item...)
			itemIdx++
		} else if err := readProducerStatus(producerStatus); err != nil {
			return fmt.Errorf("producer failed for %s: %w", path, err)
		}

		// if we have collected enough items or the channel is closed, upload the part
		if itemIdx%s.MaxItemsPerPartUpload == 0 || !ok {
			partInput := &s3.UploadPartInput{
				Body:       bytes.NewReader(outputBuffer),
				Bucket:     resp.Bucket,
				Key:        resp.Key,
				PartNumber: aws.Int32(partNumber),
				UploadId:   resp.UploadId,
			}
			uploadResult, err := s.Client.UploadPart(ctx, partInput)
			if err != nil {
				return fmt.Errorf("upload part %d for %s: %w", partNumber, path, err)
			}
			o11y.S(ctx).Infof("Uploaded part %d to path %s", partNumber, path)
			completedParts = append(completedParts, types.CompletedPart{
				ETag:       uploadResult.ETag,
				PartNumber: aws.Int32(partNumber),
			})
			outputBuffer = []byte{}
			partNumber += 1
		}

		if !ok {
			break
		}
	}

	var parts []string
	for _, part := range completedParts {
		parts = append(parts, fmt.Sprintf("{%d: %s}", *part.PartNumber, *part.ETag))
	}
	o11y.S(ctx).Infof("Completed uploading parts: %s", strings.Join(parts, ", "))

	compInput := &s3.CompleteMultipartUploadInput{
		Bucket:   resp.Bucket,
		Key:      resp.Key,
		UploadId: resp.UploadId,
		MultipartUpload: &types.CompletedMultipartUpload{
			Parts: completedParts,
		},
	}
	_, err = s.Client.CompleteMultipartUpload(ctx, compInput)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Failed to complete multipart upload, aborting")
		return fmt.Errorf("complete multipart upload for %s: %w", path, err)
	}

	completed = true
	return nil
}

func (s *S3Writer) Persist(path string, data []byte) error {
	ctx := context.Background()
	_, err := s.Client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: &s.Bucket,
		Key:    &path,
		Body:   bytes.NewReader(data),
	})
	if err != nil {
		return fmt.Errorf("put object %s: %w", path, err)
	}
	return nil
}

type S3Reader struct {
	Client *s3.Client
	Bucket string
}

func (s *S3Reader) GetTimeOfLastExport(ctx context.Context, exportPath string) (*int64, error) {
	lastExportTime := int64(0)
	timestampsPrefix := fmt.Sprintf("%s/%s/", exportPath, TimestampsFolder)

	input := &s3.ListObjectsV2Input{
		Bucket: &s.Bucket,
		Prefix: &timestampsPrefix,
	}

	for {
		result, err := s.Client.ListObjectsV2(ctx, input)
		if err != nil {
			return nil, err
		}

		for _, object := range result.Contents {
			key := *object.Key
			filename := filepath.Base(key)
			timestampStr := strings.Split(filename, "_")[0]
			unixTime, err := strconv.ParseInt(timestampStr, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse timestamp: %v", err)
			}
			lastExportTime = max(lastExportTime, unixTime)
		}

		if *result.IsTruncated {
			input.ContinuationToken = result.NextContinuationToken
		} else {
			break
		}
	}

	if lastExportTime == 0 {
		return nil, errors.New("no exports found")
	}

	return &lastExportTime, nil
}

func NewS3Reader(ctx context.Context, bucket string, region, endpoint string) (*S3Reader, error) {
	o11y.S(ctx).Infof("Creating new S3 reader")
	awsCfg := awsConfig.LoadAWSDefaultConfig(context.Background(), region, endpoint)
	client := s3.NewFromConfig(awsCfg)
	o11y.S(ctx).Infof("Created new S3 reader")
	return &S3Reader{
		Client: client,
		Bucket: bucket,
	}, nil
}

func NewS3Writer(ctx context.Context, bucket, region, endpoint string, maxItemsPerPartUpload int) (*S3Writer, error) {
	o11y.S(ctx).Infof("Creating new S3 writer")
	awsCfg := awsConfig.LoadAWSDefaultConfig(context.Background(), region, endpoint)
	client := s3.NewFromConfig(awsCfg)
	o11y.S(ctx).Infof("Created new S3 writer")
	return &S3Writer{
		Client:                client,
		Bucket:                bucket,
		MaxItemsPerPartUpload: maxItemsPerPartUpload,
	}, nil
}
