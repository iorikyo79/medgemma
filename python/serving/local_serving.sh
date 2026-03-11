#!/bin/bash
# MedGemma Local Serving Script
# Runs the Triton + Gunicorn serving stack locally without Vertex AI

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/models/medgemma-4b-it"
HTTP_PORT=8080
MODEL_REST_PORT=8600
TRITON_PORT=8500
TRITON_CONTAINER_NAME="triton-medgemma"
SHARED_MEMORY_SIZE="16g"

# Parse command line arguments
COMMAND=${1:-"start"}

case $COMMAND in
  start)
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}MedGemma Local Serving${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""

    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
        exit 1
    fi

    echo -e "${GREEN}Model path:${NC} $MODEL_PATH"
    echo -e "${GREEN}HTTP Port:${NC} $HTTP_PORT"
    echo -e "${GREEN}Model REST Port:${NC} $MODEL_REST_PORT"
    echo ""

    # Check if Triton container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${TRITON_CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Triton container already exists. Removing...${NC}"
        docker stop "${TRITON_CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${TRITON_CONTAINER_NAME}" 2>/dev/null || true
    fi

    # Create model repository structure for Triton
    echo -e "${BLUE}[1/4] Creating Triton model repository...${NC}"
    REPO_DIR="/tmp/medgemma_triton_repo"
    rm -rf "$REPO_DIR"
    mkdir -p "$REPO_DIR/default/1"

    # Create model config for vLLM
    # Note: Use container path (/model) since MODEL_PATH is mounted to /model in the container
    cat > "$REPO_DIR/default/1/model.json" <<EOF
{
  "model": "/model",
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.85,
  "max_num_seqs": 8,
  "disable_log_stats": true,
  "max_model_len": 2048,
  "trust_remote_code": true
}
EOF

    # Create config.pbtxt
    cat > "$REPO_DIR/default/config.pbtxt" <<'EOF'
backend: "vllm"
instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]
EOF

    echo -e "${GREEN}Model repository created at $REPO_DIR${NC}"
    echo ""

    # Start Triton server
    echo -e "${BLUE}[2/4] Starting Triton Inference Server...${NC}"
    docker run -d \
      --name "${TRITON_CONTAINER_NAME}" \
      --gpus all \
      --shm-size="${SHARED_MEMORY_SIZE}" \
      -p "${TRITON_PORT}":8500 \
      -p "${MODEL_REST_PORT}":8000 \
      -v "${MODEL_PATH}:/model:ro" \
      -v "${REPO_DIR}:/model_repository:ro" \
      nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 \
      tritonserver \
      --model-repository=/model_repository \
      --allow-grpc=true \
      --grpc-address=0.0.0.0 \
      --grpc-port=8500 \
      --allow-http=true \
      --http-address=0.0.0.0 \
      --http-port=8000 \
      --strict-readiness=true

    echo -e "${GREEN}Triton container started${NC}"
    echo ""

    # Wait for Triton to be ready
    echo -e "${BLUE}[3/4] Waiting for Triton server to be ready...${NC}"
    MAX_ATTEMPTS=60
    ATTEMPT=0
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if docker exec "${TRITON_CONTAINER_NAME}" curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
            echo -e "${GREEN}Triton server is ready!${NC}"
            break
        fi
        ATTEMPT=$((ATTEMPT + 1))
        echo -n "."
        sleep 2
    done
    echo ""

    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        echo -e "${RED}Triton server failed to start. Check logs with:${NC}"
        echo "docker logs ${TRITON_CONTAINER_NAME}"
        exit 1
    fi

    # Set environment variables for Gunicorn
    echo -e "${BLUE}[4/4] Starting Gunicorn API server...${NC}"
    export AIP_HTTP_PORT="$HTTP_PORT"
    export AIP_PREDICT_ROUTE="/predict"
    export AIP_HEALTH_ROUTE="/health"
    export ENABLE_CLOUD_LOGGING="false"
    export MODEL_REST_PORT="$MODEL_REST_PORT"
    export LOCAL_MODEL_PATH="$MODEL_PATH"
    export IS_DEBUGGING="true"

    # Change to serving directory and set PYTHONPATH
    cd "$(dirname "$0")"
    export PYTHONPATH="$(dirname "$0")/..:$PYTHONPATH"

    # Start Gunicorn server in foreground
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}Server Ready!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "API Server: ${BLUE}http://localhost:${HTTP_PORT}${NC}"
    echo -e "Health Check: ${BLUE}http://localhost:${HTTP_PORT}/health${NC}"
    echo -e "Predict Endpoint: ${BLUE}http://localhost:${HTTP_PORT}/predict${NC}"
    echo ""
    echo -e "Triton Server: ${BLUE}http://localhost:${MODEL_REST_PORT}${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
    echo ""

    # Trap to cleanup on exit
    trap 'echo -e "\n${YELLOW}Shutting down...${NC}"; docker stop "${TRITON_CONTAINER_NAME}" 2>/dev/null || true; exit 0' INT TERM

    # Start Gunicorn (will block until interrupted)
    python3 server_gunicorn.py \
      --local_model_path="$MODEL_PATH" \
      --alsologtostderr \
      --verbosity=1
    ;;

  stop)
    echo -e "${YELLOW}Stopping MedGemma local serving...${NC}"

    # Stop Triton container
    if docker ps --format '{{.Names}}' | grep -q "^${TRITON_CONTAINER_NAME}$"; then
        echo "Stopping Triton container..."
        docker stop "${TRITON_CONTAINER_NAME}"
        docker rm "${TRITON_CONTAINER_NAME}"
        echo -e "${GREEN}Triton container stopped${NC}"
    else
        echo "Triton container not running"
    fi

    # Kill any running gunicorn processes
    pkill -f "serving.server_gunicorn" 2>/dev/null && echo "Gunicorn server stopped" || echo "Gunicorn server not running"

    # Cleanup temp files
    rm -rf /tmp/medgemma_triton_repo

    echo -e "${GREEN}All servers stopped${NC}"
    ;;

  status)
    echo -e "${BLUE}MedGemma Local Serving Status${NC}"
    echo ""

    # Check Triton
    if docker ps --format '{{.Names}}' | grep -q "^${TRITON_CONTAINER_NAME}$"; then
        echo -e "Triton Server: ${GREEN}Running${NC}"
        docker ps --filter "name=${TRITON_CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        echo -e "Triton Server: ${RED}Not Running${NC}"
    fi
    echo ""

    # Check Gunicorn
    if pgrep -f "gunicorn" > /dev/null && netstat -tuln 2>/dev/null | grep -q ":${HTTP_PORT} "; then
        echo -e "Gunicorn Server: ${GREEN}Running${NC} (PID: $(pgrep -f 'gunicorn' | head -1))"
    else
        echo -e "Gunicorn Server: ${RED}Not Running${NC}"
    fi
    echo ""

    # Test API if running
    if pgrep -f "serving.server_gunicorn" > /dev/null; then
        echo "Testing API..."
        if curl -s "http://localhost:${HTTP_PORT}/health" > /dev/null 2>&1; then
            echo -e "API Health Check: ${GREEN}OK${NC}"
            curl -s "http://localhost:${HTTP_PORT}/health" | python3 -m json.tool 2>/dev/null || true
        else
            echo -e "API Health Check: ${RED}Failed${NC}"
        fi
    fi
    ;;

  logs)
    if [ -n "${2:-}" ]; then
        case "$2" in
          triton)
            docker logs -f "${TRITON_CONTAINER_NAME}" 2>/dev/null || echo "Triton logs not available"
            ;;
          gunicorn)
            echo "Gunicorn logs (check above output)"
            ;;
          *)
            echo "Usage: $0 logs [triton|gunicorn]"
            exit 1
            ;;
        esac
    else
        echo "Usage: $0 logs [triton|gunicorn]"
    fi
    ;;

  test)
    # Run curl tests
    if [ ! -f "./test_curl.sh" ]; then
        echo -e "${RED}test_curl.sh not found${NC}"
        exit 1
    fi
    ./test_curl.sh
    ;;

  *)
    echo "MedGemma Local Serving Script"
    echo ""
    echo "Usage: $0 {start|stop|status|logs|test}"
    echo ""
    echo "Commands:"
    echo "  start    - Start Triton and Gunicorn servers"
    echo "  stop     - Stop all servers"
    echo "  status   - Show server status"
    echo "  logs     - Show logs [triton|gunicorn]"
    echo "  test     - Run API tests"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start all servers"
    echo "  $0 status         # Check status"
    echo "  $0 logs triton    # View Triton logs"
    echo "  $0 stop           # Stop all servers"
    exit 1
    ;;
esac
