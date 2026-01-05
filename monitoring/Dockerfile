FROM prom/prometheus:latest

# Switch to root to install packages
USER root

# Install gettext for envsubst (environment variable substitution)
RUN apk add --no-cache gettext

# Copy configuration template and entrypoint
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Expose Prometheus port
EXPOSE 9090

# Switch back to nobody user for security
USER nobody

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]