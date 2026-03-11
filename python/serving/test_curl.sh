#!/bin/bash
# MedGemma Server Test Script using curl
# Server URL
SERVER_URL="http://localhost:8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "MedGemma Server Test Suite (curl)"
echo "============================================================"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}[Test 1] Health Check${NC}"
echo "GET $SERVER_URL/health"
echo ""
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" "$SERVER_URL/health")
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
else
    echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
    echo "$body"
fi
echo ""
echo "------------------------------------------------------------"
echo ""

# Test 2: Simple Predict (text only)
echo -e "${YELLOW}[Test 2] Simple Predict Endpoint${NC}"
echo "POST $SERVER_URL/predict"
echo "Request: What is medical imaging?"
echo ""
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "$SERVER_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is medical imaging? Answer in one sentence.",
    "max_tokens": 100,
    "temperature": 0.7
  }')
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
else
    echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
    echo "$body"
fi
echo ""
echo "------------------------------------------------------------"
echo ""

# Test 3: Chat Completions (text only)
echo -e "${YELLOW}[Test 3] Chat Completions (text only)${NC}"
echo "POST $SERVER_URL/v1/chat/completions"
echo "Question: What is a chest X-ray used for?"
echo ""
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is a chest X-ray used for? Answer briefly."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }')
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null | grep -A 20 '"content"' || echo "$body"
    echo ""
    echo "Token Usage:"
    echo "$body" | python3 -m json.tool 2>/dev/null | grep -A 5 '"usage"' || echo "$body"
else
    echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
    echo "$body"
fi
echo ""
echo "------------------------------------------------------------"
echo ""

# Test 4: Multi-turn Conversation
echo -e "${YELLOW}[Test 4] Multi-turn Conversation${NC}"
echo "POST $SERVER_URL/v1/chat/completions"
echo "Conversation about CT vs MRI"
echo ""
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is a CT scan?"},
      {"role": "assistant", "content": "A CT scan uses X-rays to create detailed images of the body."},
      {"role": "user", "content": "How is it different from MRI? Answer in 2-3 sentences."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }')
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null | grep -A 30 '"content"' || echo "$body"
else
    echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
    echo "$body"
fi
echo ""
echo "------------------------------------------------------------"
echo ""

# Test 5: Chat with Different Parameters
echo -e "${YELLOW}[Test 5] Chat with Low Temperature (more deterministic)${NC}"
echo "POST $SERVER_URL/v1/chat/completions"
echo "Question: List 3 common medical imaging modalities"
echo ""
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "List 3 common medical imaging modalities."}
    ],
    "max_tokens": 100,
    "temperature": 0.3,
    "top_p": 0.9
  }')
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" -eq 200 ]; then
    echo -e "${GREEN}✓ PASSED (HTTP $http_code)${NC}"
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null | grep -A 20 '"content"' || echo "$body"
else
    echo -e "${RED}✗ FAILED (HTTP $http_code)${NC}"
    echo "$body"
fi
echo ""
echo "------------------------------------------------------------"
echo ""

# Summary
echo "============================================================"
echo "Test Suite Complete!"
echo "============================================================"
echo ""
echo "Server: $SERVER_URL"
echo ""
echo "Quick Test Commands:"
echo "  # Health check"
echo "  curl $SERVER_URL/health"
echo ""
echo "  # Simple predict"
echo '  curl -X POST '"$SERVER_URL/predict"' -H "Content-Type: application/json" -d '"'"'{"prompt": "Hello", "max_tokens": 50}'"'"
echo ""
echo "  # Chat completion"
echo '  curl -X POST '"$SERVER_URL/v1/chat/completions"' -H "Content-Type: application/json" -d '"'"'{"messages": [{"role": "user", "content": "Hello"}]}'"'"
echo ""
