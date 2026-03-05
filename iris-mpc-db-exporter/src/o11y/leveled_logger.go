package o11y

import (
	"go.uber.org/zap"
)

// RetryableHttpLeveledLogger is an implementation of the retryablehttp.LeveledLogger interface used by retryablehttp.Client
type RetryableHttpLeveledLogger struct{}

func NewRetryableHttpLeveledLogger() *RetryableHttpLeveledLogger {
	return &RetryableHttpLeveledLogger{}
}

func (l *RetryableHttpLeveledLogger) Error(msg string, keysAndValues ...interface{}) {
	zap.S().With(keysAndValues...).Error(msg)
}

func (l *RetryableHttpLeveledLogger) Info(msg string, keysAndValues ...interface{}) {
	zap.S().With(keysAndValues...).Info(msg)
}

func (l *RetryableHttpLeveledLogger) Debug(msg string, keysAndValues ...interface{}) {
	zap.S().With(keysAndValues...).Debug(msg)
}

func (l *RetryableHttpLeveledLogger) Warn(msg string, keysAndValues ...interface{}) {
	zap.S().With(keysAndValues...).Warn(msg)
}
