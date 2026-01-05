FROM prom/prometheus:latest

# Install envsubst for environment variable substitution
USER root
RUN apk add --no-cache gettext

# Copy configuration files
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Expose Prometheus port
EXPOSE 9090

# Switch back to prometheus user
USER nobody

# Run entrypoint script
ENTRYPOINT ["/entrypoint.sh"]