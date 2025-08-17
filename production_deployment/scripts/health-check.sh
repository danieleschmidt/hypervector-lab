#!/bin/bash
set -euo pipefail

# HyperVector-Lab Health Check Script

echo "🏥 Running health check..."

# Check Python environment
python -c "import hypervector; print('✅ Python import successful')"

# Check system resources
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

echo "📊 System metrics:"
echo "  Memory usage: ${MEMORY_USAGE}%"
echo "  CPU usage: ${CPU_USAGE}%"

# Check for common issues
if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "⚠️ High memory usage detected"
fi

if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "⚠️ High CPU usage detected"
fi

echo "✅ Health check completed"
