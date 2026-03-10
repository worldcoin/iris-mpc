package o11y

import (
	"context"
	"fmt"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/DataDog/dd-trace-go.v1/ddtrace"
	"gopkg.in/DataDog/dd-trace-go.v1/ddtrace/tracer"
)

func S(ctx context.Context) *zap.SugaredLogger {
	if span, ok := tracer.SpanFromContext(ctx); ok {
		return zap.S().With(
			"dd.trace_id", span.Context().TraceID(),
			"dd.span_id", span.Context().SpanID(),
		)
	}
	return zap.S()
}

func ConfigureLogging() {
	cfg := zap.Config{
		Level:       zap.NewAtomicLevelAt(zap.InfoLevel),
		Development: false,
		Sampling: &zap.SamplingConfig{
			Initial:    100,
			Thereafter: 100,
		},
		Encoding: "json",
		EncoderConfig: zapcore.EncoderConfig{
			TimeKey:     "ts",
			LevelKey:    "level",
			MessageKey:  "msg",
			LineEnding:  zapcore.DefaultLineEnding,
			EncodeLevel: zapcore.CapitalLevelEncoder,
			EncodeTime:  zapcore.ISO8601TimeEncoder,
		},
		OutputPaths:      []string{"stdout"},
		ErrorOutputPaths: []string{"stdout"},
	}

	zapLogger, err := cfg.Build()
	if err != nil {
		panic(fmt.Errorf("logger building error: %s", err.Error()))
	}
	zap.ReplaceGlobals(zapLogger)
}

func SpanFromTrace(ctx context.Context, trace map[string]string, spanName string) (ddtrace.Span, context.Context) {
	extractedCtx, err := tracer.Extract(tracer.TextMapCarrier(trace))
	if err != nil {
		S(ctx).With(zap.Error(err)).Error("could not extract ddHeaders")
		return tracer.StartSpanFromContext(ctx, spanName)
	}

	return tracer.StartSpanFromContext(ctx, spanName, tracer.ChildOf(extractedCtx))
}
