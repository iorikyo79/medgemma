#!/bin/bash
# MedGemma Server Test Script with Image (curl)
# Server URL
SERVER_URL="http://localhost:8080"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test image path
TEST_IMAGE="/home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/testdata/image.jpeg"

echo "============================================================"
echo "MedGemma Image Test Suite (curl)"
echo "============================================================"
echo ""

# Check if test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${RED}Error: Test image not found at $TEST_IMAGE${NC}"
    echo "Please provide a valid image path."
    exit 1
fi

echo "Test Image: $TEST_IMAGE"
echo ""

# Test: Chat with Image
echo -e "${YELLOW}[Test] Chat Completions with Image${NC}"
echo "POST $SERVER_URL/v1/chat/completions"
echo "Question: What do you see in this image?"
echo ""

# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 "$TEST_IMAGE")

response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"What do you see in this image? Please describe it.\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMAGE_BASE64\"}}
        ]
      }
    ],
    \"max_tokens\": 300,
    \"temperature\": 0.7
  }")

http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
    echo ""
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null | grep -A 50 '"content"' || echo "$body"
else
    echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
    echo "$body"
fi
echo ""
echo "------------------------------------------------------------"
echo ""

# Test: Custom Image from Command Line Argument
if [ -n "$1" ] && [ -f "$1" ]; then
    echo -e "${YELLOW}[Test] Chat with Custom Image${NC}"
    echo "Image: $1"
    echo ""

    CUSTOM_IMAGE_BASE64=$(base64 -w 0 "$1")

    response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "$SERVER_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"messages\": [
          {
            \"role\": \"user\",
            \"content\": [
              {\"type\": \"text\", \"text\": \"Analyze this medical image.\"},
              {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$CUSTOM_IMAGE_BASE64\"}}
            ]
          }
        ],
        \"max_tokens\": 300,
        \"temperature\": 0.7
      }")

    http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_CODE/d')

    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
        echo ""
        echo "Response:"
        echo "$body" | python3 -m json.tool 2>/dev/null | grep -A 50 '"content"' || echo "$body"
    else
        echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
        echo "$body"
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo ""
fi

echo "============================================================"
echo "Image Test Complete!"
echo "============================================================"
echo ""
echo "Usage with custom image:"
echo "  ./test_curl_with_image.sh /path/to/your/image.jpg"
echo ""
