#!/bin/sh

echo "🚀 Starting Prometheus..."
echo "📊 API URL: ${API_URL}"
echo "🔗 Grafana Cloud:  ${GRAFANA_CLOUD_URL}"

# Replace environment variables in template
envsubst < /etc/prometheus/prometheus.yml.template > /etc/prometheus/prometheus.yml

echo "✅ Configuration ready"

# Start Prometheus
exec /bin/prometheus \
  --config. file=/etc/prometheus/prometheus. yml \
  --storage. tsdb.path=/prometheus \
  --storage.tsdb.retention.time=15d \
  --web.listen-address=:${PORT:-9090} \
  --web.enable-lifecycle