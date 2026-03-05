package converter

import (
	"encoding/binary"
	"fmt"

	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
)

type BinaryConverter struct {
	SingleRowTotalSize  int
	SingleCodeSize      int
	SingleMaskSize      int
	SingleIdSize        int
	SingleVersionIdSize int
}

func NewBinaryConverter(codeSize, maskSize, idSize, versionIdSize int) *BinaryConverter {
	return &BinaryConverter{
		SingleCodeSize:      codeSize,
		SingleMaskSize:      maskSize,
		SingleIdSize:        idSize,
		SingleVersionIdSize: versionIdSize,
		SingleRowTotalSize:  (codeSize * 2) + (maskSize * 2) + idSize + versionIdSize,
	}
}

func (c *BinaryConverter) validateStoredIris(iris iris.StoredIris) error {
	validate := func(data []byte, name string, expectedSize int) error {
		if len(data) != expectedSize {
			return fmt.Errorf("invalid %s size, expected %d bytes, got %d, (ID: %d)", name, expectedSize, len(data), iris.ID)
		}
		return nil
	}

	if err := validate(iris.LeftCode, "left code", c.SingleCodeSize); err != nil {
		return err
	}

	if err := validate(iris.LeftMask, "left mask", c.SingleMaskSize); err != nil {
		return err
	}

	if err := validate(iris.RightCode, "right code", c.SingleCodeSize); err != nil {
		return err
	}

	if err := validate(iris.RightMask, "right mask", c.SingleMaskSize); err != nil {
		return err
	}

	return nil
}

func (c *BinaryConverter) GetExtension() string {
	return "bin"
}

func (c *BinaryConverter) Convert(data []iris.StoredIris) ([]byte, error) {
	outputArray := make([]byte, c.SingleRowTotalSize*len(data))

	for i, item := range data {
		if err := c.validateStoredIris(item); err != nil {
			return nil, err
		}

		serialIdBytes := make([]byte, c.SingleIdSize)
		binary.BigEndian.PutUint32(serialIdBytes, uint32(item.ID))

		// Step 1: Write serial id
		// Starting offset for this row:
		start := i * c.SingleRowTotalSize
		copy(outputArray[start:start+c.SingleIdSize], serialIdBytes)

		// Step 2: Write masks and codes as even odd pairs
		// Offset points where we left off in this row:
		offset := start + c.SingleIdSize
		offset = storeAsEvenOddPairs(item.LeftCode, outputArray, offset)
		offset = storeAsEvenOddPairs(item.LeftMask, outputArray, offset)
		offset = storeAsEvenOddPairs(item.RightCode, outputArray, offset)
		offset = storeAsEvenOddPairs(item.RightMask, outputArray, offset)

		// Step 3: Write version id
		versionIdBytes := make([]byte, c.SingleVersionIdSize)
		binary.BigEndian.PutUint16(versionIdBytes, uint16(item.VersionID))
		copy(outputArray[offset:offset+c.SingleVersionIdSize], versionIdBytes)
	}

	return outputArray, nil
}

func (c *BinaryConverter) ConvertSingle(item iris.StoredIris) ([]byte, error) {
	outputArray := make([]byte, c.SingleRowTotalSize)

	if err := c.validateStoredIris(item); err != nil {
		return nil, err
	}

	// Step 1: Write serial id
	serialIdBytes := make([]byte, c.SingleIdSize)
	binary.BigEndian.PutUint32(serialIdBytes, uint32(item.ID))
	copy(outputArray[0:c.SingleIdSize], serialIdBytes)

	// Step 2: Write masks and codes as even odd pairs
	// Offsets for subsequent data, observe that the helper is mutating the array in place
	offset := c.SingleIdSize
	offset = storeAsEvenOddPairs(item.LeftCode, outputArray, offset)
	offset = storeAsEvenOddPairs(item.LeftMask, outputArray, offset)
	offset = storeAsEvenOddPairs(item.RightCode, outputArray, offset)
	offset = storeAsEvenOddPairs(item.RightMask, outputArray, offset)

	// Step 3: Write version id
	versionIdBytes := make([]byte, c.SingleVersionIdSize)
	binary.BigEndian.PutUint16(versionIdBytes, uint16(item.VersionID))
	copy(outputArray[offset:offset+c.SingleVersionIdSize], versionIdBytes)

	return outputArray, nil
}

func storeAsEvenOddPairs(input []byte, output []byte, offset int) int {
	if len(input)%2 != 0 {
		panic("input must have an even number of bytes")
	}

	for i := 0; i < len(input); i += 2 {
		even := input[i]
		odd := input[i+1]

		// ---- SHIFT BY 128 ----
		// We treat these shifts as (value - 128). Because `byte` in Go is unsigned,
		// storing a negative value or > 255 results in wrap-around (mod 256).
		// This is analogous to casting in Rust with `(x as i8) - 128`, then storing as `u8`.
		shiftEven := int(even) - 128
		shiftOdd := int(odd) - 128

		// Write back into output in odd and even blocks
		// We are storing the even bytes first in the output
		evenIndex := offset + i/2
		output[evenIndex] = byte(shiftEven)

		// We are storing the odd bytes after the even ones in the output
		oddIndex := len(input)/2 + offset + i/2
		output[oddIndex] = byte(shiftOdd)

		// final output should be:
		// byte[] = [even0, even1, even2, ..., evenN, odd0, odd1, odd2, ..., oddN]
	}

	offset += len(input)

	return offset
}
