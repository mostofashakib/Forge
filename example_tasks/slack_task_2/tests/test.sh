#!/bin/bash
set -euo pipefail

python3 -m rewardkit /tests \
  --workspace /app \
  --output /logs/verifier/reward.json
