package persistence

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"go.uber.org/zap"

	awsConfig "github.com/worldcoin/iris-mpc-db-exporter/src/aws"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
)

type S3Writer struct {
	Client                *s3.Client
	Bucket                string
	MaxItemsPerPartUpload int
}

func (s *S3Writer) PersistStream(ctx context.Context, path string, inputChannel <-chan []byte) error {
	input := &s3.CreateMultipartUploadInput{
		Bucket: &s.Bucket,
		Key:    &path,
	}
	resp, err := s.Client.CreateMultipartUpload(ctx, input)
	if err != nil {
		return err
	}

	o11y.S(ctx).Infof("Created multipart upload with ID %s", *resp.UploadId)

	var completedParts []types.CompletedPart
	var outputBuffer []byte

	partNumber := int32(1)
	itemIdx := 0

	for {
		item, ok := <-inputChannel
		if ok {
			outputBuffer = append(outputBuffer, item...)
			itemIdx++
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
				aboInput := &s3.AbortMultipartUploadInput{
					Bucket:   resp.Bucket,
					Key:      resp.Key,
					UploadId: resp.UploadId,
				}
				_, aboErr := s.Client.AbortMultipartUpload(ctx, aboInput)
				if aboErr != nil {
					return aboErr
				}
				return err
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
		abortInput := &s3.AbortMultipartUploadInput{
			Bucket:   resp.Bucket,
			Key:      resp.Key,
			UploadId: resp.UploadId,
		}
		_, abortErr := s.Client.AbortMultipartUpload(ctx, abortInput)
		if abortErr != nil {
			return abortErr
		}
		return err
	}

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
		log.Fatal(err)
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
