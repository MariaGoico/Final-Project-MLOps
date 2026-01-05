FROM prom/prometheus:latest
EXPOSE 9090
CMD ["--config.file=/etc/secrets/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.listen-address=:9090"]