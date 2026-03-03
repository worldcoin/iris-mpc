package commands

import (
	"context"
	"crypto/rand"
	mathrand "math/rand"

	"go.uber.org/zap"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
)

func generateIrisCode(ctx context.Context, size int) []byte {
	return generateRandomCode(ctx, size)
}

func generateIrisMask(ctx context.Context, size int) []byte {
	return generateRandomCode(ctx, size)
}

func generateVersionID() int16 {
	return int16(mathrand.Intn(100))
}

func generateRandomCode(ctx context.Context, size int) []byte {
	randomBytes := make([]byte, size)

	_, err := rand.Read(randomBytes)
	if err != nil {
		o11y.S(ctx).Fatalf("Error generating random bytes: %v", err)
		return nil
	}

	return randomBytes
}

func PopulateDbWithMockDataCommand(ctx context.Context, store iris.Store, count int) {
	cfg := config.Load()
	err := store.CreateTable(ctx)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Fatalf("Error creating irises table: %v", err)
	}

	for i := 0; i < count; i++ {
		mockIris := iris.StoredIris{
			LeftCode:  generateIrisCode(ctx, cfg.SingleCodeSize),
			LeftMask:  generateIrisMask(ctx, cfg.SingleMaskSize),
			RightCode: generateIrisCode(ctx, cfg.SingleCodeSize),
			RightMask: generateIrisMask(ctx, cfg.SingleMaskSize),
			VersionID: generateVersionID(),
		}

		err := store.InsertIris(ctx, mockIris)

		if err != nil {
			o11y.S(ctx).With(zap.Error(err)).Fatalf("Error inserting iris: %v", err)
		}
	}

	o11y.S(ctx).Infof("Inserted %d irises", count)
}
