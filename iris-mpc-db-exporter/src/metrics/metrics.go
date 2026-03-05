package metrics

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/DataDog/datadog-go/v5/statsd"
	"go.uber.org/zap"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
)

var Client *statsd.Client

func InitMetrics(ctx context.Context, conf config.Config) error {
	tags := []string{fmt.Sprintf("pod_name:%s", os.Getenv("HOSTNAME"))}
	var err error
	Client, err = statsd.New(conf.DatadogHost, statsd.WithNamespace(conf.ServiceName), statsd.WithTags(tags))
	o11y.S(ctx).Infof("initializing metrics client - DatadogHost: %s, ServiceName: %s", conf.DatadogHost, conf.ServiceName)
	if err != nil {
		return fmt.Errorf("error creating statsd client: %w", err)
	}
	return nil
}

func CloseMetrics(ctx context.Context, sleepDuration time.Duration) {
	if Client != nil {
		err := Client.Close()
		if err != nil {
			o11y.S(ctx).Errorf("error closing metrics client: %v", err)
		} else {
			o11y.S(ctx).Infof("metrics client closed gracefully. sleeping for %v", sleepDuration)
		}

		// Small delay to allow kernel to transmit UDP packets before process/container exits.
		// This is necessary because UDP is fire-and-forget, and the kernel may not have
		// transmitted the packet before the container's network namespace is torn down.
		time.Sleep(sleepDuration)
		o11y.S(ctx).Info("sleep during metrics client close finished")
	}
}

func MetricIncrement(ctx context.Context, name string, tags []string, rate float64) {
	err := Client.Incr(name, tags, rate)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Error incrementing statsd metric")
	}
}
