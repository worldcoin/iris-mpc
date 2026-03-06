package aws

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsConfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"go.uber.org/zap"

	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
)

func LoadAWSDefaultConfig(ctx context.Context, region, endpoint string) aws.Config {
	options := []func(*awsConfig.LoadOptions) error{
		awsConfig.WithRegion(region),
	}

	if endpoint != "" {
		endpointResolver := awsConfig.WithEndpointResolverWithOptions(aws.EndpointResolverWithOptionsFunc(func(_, _ string, _ ...interface{}) (aws.Endpoint, error) {
			return aws.Endpoint{
				URL:               endpoint,
				PartitionID:       "aws",
				SigningRegion:     region,
				HostnameImmutable: true,
			}, nil
		}))
		options = append(options, endpointResolver)
		options = append(options, awsConfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("aws", "aws", "aws")))
	}

	awsCfg, err := awsConfig.LoadDefaultConfig(ctx, options...)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Fatal("unable to load AWS SDK config")
	}
	return awsCfg
}
