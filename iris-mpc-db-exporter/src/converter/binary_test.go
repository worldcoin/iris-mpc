package converter

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
)

// TestBinaryConverter_ConvertSingle verifies the following:
//  1. Output array size is correct (SingleRowTotalSize).
//  2. ID is placed in the first 4 bytes in BigEndian format.
//  3. Even-odd rearrangement is correct for code and mask data.
func TestBinaryConverter_ConvertSingle(t *testing.T) {
	c := NewBinaryConverter(8, 4, 4, 2)

	// Build small test data for code/mask
	leftCode := []byte{128, 129, 130, 131, 132, 133, 134, 135} // length=8, must be even
	leftMask := []byte{140, 141, 142, 143}                     // length=4, must be even

	rightCode := []byte{150, 151, 152, 153, 154, 155, 156, 157} // length=8, must be even
	rightMask := []byte{160, 161, 162, 163}                     // length=4, must be even

	item := iris.StoredIris{
		ID:        42,
		LeftCode:  leftCode,
		LeftMask:  leftMask,
		RightCode: rightCode,
		RightMask: rightMask,
		VersionID: 55,
	}

	output, err := c.ConvertSingle(item)
	assert.NoError(t, err)

	// Verify the total length matches the formula
	expectedLength := c.SingleRowTotalSize
	if len(output) != expectedLength {
		t.Fatalf("expected output length %d, got %d", expectedLength, len(output))
	}

	// ID check: first 4 bytes in BigEndian
	gotID := binary.BigEndian.Uint32(output[0:c.SingleIdSize])
	expectedID := uint32(item.ID)
	if gotID != expectedID {
		t.Errorf("expected ID %d, got %d", expectedID, gotID)
	}

	// Because SingleCodeSize=25600, SingleMaskSize=12800 for real usage,
	// in the test we only wrote 4 bytes each. Let's verify that these 4 bytes
	// are placed correctly in even-odd rearranged form.
	// Offsets:
	//  0-3 (4 bytes) => ID
	//  next up => leftCode rearranged, leftMask rearranged, etc.
	offset := c.SingleIdSize

	// We expect the storeAsEvenOddPairs logic to place:
	//   even bytes of leftCode first, then odd bytes of leftCode in the second half, etc.

	// For leftCode = {128, 129, 130, 131},
	//  shiftEven for 128 = 128 - 128 = 0
	//  shiftOdd for 129  = 129 - 128 = 1
	//  shiftEven for 130 = 130 - 128 = 2
	//  shiftOdd for 131  = 131 - 128 = 3
	//
	// The final array portion for leftCode should be:
	//  [0, 2, 1, 3] in the even and odd blocks, but arranged as:
	//   evens in the first half: {0, 2}
	//   odds in the second half: {1, 3}

	// We'll just spot-check that the first two bytes after offset hold the two "evens"
	// and the next two bytes after that hold the "odds".
	// But remember the function merges them with the next calls. We have to isolate
	// how many bytes each call writes. Each call writes len(input) bytes, but in
	// an even-odd pattern. So for 4 bytes of input, the function will produce 4 bytes
	// in the output, placed in some partition of the final slice.

	// For simplicity, let's test the entire ConvertSingle result in smaller blocks.

	// Extract the portion for leftCode. storeAsEvenOddPairs writes 4 bytes total
	leftCodeSize := len(leftCode)
	leftCodeOutput := output[offset : offset+leftCodeSize]
	offset += leftCodeSize

	// Extract the portion for leftMask. storeAsEvenOddPairs writes 4 bytes total
	leftMaskSize := len(leftMask)
	leftMaskOutput := output[offset : offset+leftMaskSize]
	offset += leftMaskSize

	// Extract the portion for rightCode.
	rightCodeSize := len(rightCode)
	rightCodeOutput := output[offset : offset+rightCodeSize]
	offset += rightCodeSize

	// Extract the portion for rightMask.
	rightMaskSize := len(rightMask)
	rightMaskOutput := output[offset : offset+rightMaskSize]
	offset += rightMaskSize

	// Check the rearranged values for leftCodeOutput
	wantLeftCode := []byte{
		//even
		0,         // 0
		130 - 128, // 2
		132 - 128, // 4
		134 - 128, // 6
		// odd
		129 - 128, // 1
		131 - 128, // 3
		133 - 128, // 5
		135 - 128, // 7
	}
	// But note that storeAsEvenOddPairs merges the even bytes in the first half,
	// the odd bytes in the second half. The final 4 bytes from a single call
	// should appear as [0, 2, 1, 3].
	if !reflect.DeepEqual(leftCodeOutput, wantLeftCode) {
		t.Errorf("leftCodeOutput mismatch.\nGot:  %v\nWant: %v", leftCodeOutput, wantLeftCode)
	}

	// Check the rearranged values for leftMaskOutput
	wantLeftMask := []byte{
		140 - 128, // 12
		142 - 128, // 14
		141 - 128, // 13
		143 - 128, // 15
	}
	if !reflect.DeepEqual(leftMaskOutput, wantLeftMask) {
		t.Errorf("leftMaskOutput mismatch.\nGot:  %v\nWant: %v", leftMaskOutput, wantLeftMask)
	}

	// Check the rearranged values for rightCodeOutput
	wantRightCode := []byte{
		//even
		150 - 128, // 22
		152 - 128, // 24
		154 - 128, // 26
		156 - 128, // 28
		//odd
		151 - 128, // 23
		153 - 128, // 25
		155 - 128, // 27
		157 - 128, // 29
	}
	if !reflect.DeepEqual(rightCodeOutput, wantRightCode) {
		t.Errorf("rightCodeOutput mismatch.\nGot:  %v\nWant: %v", rightCodeOutput, wantRightCode)
	}

	// Check the rearranged values for rightMaskOutput
	wantRightMask := []byte{
		160 - 128, // 32
		162 - 128, // 34
		161 - 128, // 33
		163 - 128, // 35
	}
	if !reflect.DeepEqual(rightMaskOutput, wantRightMask) {
		t.Errorf("rightMaskOutput mismatch.\nGot:  %v\nWant: %v", rightMaskOutput, wantRightMask)
	}

	// Version ID check: last 2 bytes in BigEndian
	gotVersionID := binary.BigEndian.Uint16(output[offset : offset+2])
	expectedVersionID := uint16(item.VersionID)
	if gotVersionID != expectedVersionID {
		t.Errorf("expected Version ID %d, got %d", expectedVersionID, gotVersionID)
	}
}

// TestStoreAsEvenOddPairs_OddLengthInput ensures the function panics if input length is odd.
func TestStoreAsEvenOddPairs_OddLengthInput(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for odd-length input, got no panic")
		}
	}()

	// length=3 => odd. added in two steps to avoid lint error
	input := []byte{128, 129}
	input = append(input, 130)

	output := make([]byte, 6)
	offset := 0
	// This call should panic
	storeAsEvenOddPairs(input, output, offset)
}

// TestStoreAsEvenOddPairs_SimpleCase tests the even-odd rearrangement on a small slice.
func TestStoreAsEvenOddPairs_SimpleCase(t *testing.T) {
	input := []byte{128, 129, 130, 131} // length=4
	output := make([]byte, 4)
	offset := 0

	// We'll call storeAsEvenOddPairs via a testable proxy method
	finalOffset := storeAsEvenOddPairs(input, output, offset)

	// We expect finalOffset = offset + 4 (since input length is 4)
	if finalOffset != offset+4 {
		t.Errorf("expected finalOffset %d, got %d", offset+4, finalOffset)
	}

	// The rearranged result should be [even0, even1, odd0, odd1], after shift by 128
	// i.e. 128->0, 129->1, 130->2, 131->3, so final: [0, 2, 1, 3]
	want := []byte{0, 2, 1, 3}
	if !reflect.DeepEqual(output, want) {
		t.Errorf("expected output %v, got %v", want, output)
	}
}

func TestStoreAsEvenOddPairs(t *testing.T) {
	// Input: 8 bytes = 4 pairs
	// Example: 0x12 0x34 | 0xFF 0x00 | 0x88 0x99 | 0xAB 0xCD
	input := []byte{0x12, 0x34, 0xFF, 0x00, 0x88, 0x99, 0xAB, 0xCD}

	// Prepare output buffer (twice as many bytes as pairs)
	output := make([]byte, 16) // Enough to hold 4 pairs => 8 bytes
	// We’ll write starting at offset 0
	finalOffset := storeAsEvenOddPairs(input, output, 0)

	// We expect finalOffset == 8 (we processed 4 pairs => 4*2 = 8 output bytes)
	if finalOffset != 8 {
		t.Errorf("expected finalOffset = 8, got %d", finalOffset)
	}

	//0x34 = 52  =>  52 - 128 = -76 => 0xB4
	//0x12 = 18  =>  18 - 128 = -110 => 0x92
	//0x00 = 0   =>  0 - 128 = -128 => 0x80
	//0xFF = 255 =>  255 - 128 = 127 => 0x7F
	//0x99 = 153  => 153 - 128 = 25 => 0x19
	//0x88 = 136  => 136 - 128 = 8  => 0x08
	//0xCD = 205  => 205 - 128 = 77  => 0x4D
	//0xAB = 171  => 171 - 128 = 43  => 0x2B

	// even []byte{0x12 (0x92), 0xFF (0x7F), 0x88 (0x08), 0xAB (0x2B)}
	// odd []byte{0x34 (0xB4), 0x00 (0x80), 0x99 (0x19), 0xCD (0x4D)}
	// output []byte{0x92, 0x7F, 0x08, 0x2B, 0xB4, 0x80, 0x19, 0x4D}

	want := []byte{
		0x92, 0x7F,
		0x08, 0x2B,
		0xB4, 0x80,
		0x19, 0x4D}

	got := output[:finalOffset]

	if !bytes.Equal(got, want) {
		t.Errorf("storeAsEvenOddPairs mismatch.\nGot:  % X\nWant: % X", got, want)
	}
}
