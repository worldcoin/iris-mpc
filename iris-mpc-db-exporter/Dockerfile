FROM alpine:3.16

RUN apk upgrade
RUN apk add --no-cache ca-certificates tzdata

COPY iris-mpc-db-exporter /opt/iris-mpc-db-exporter
RUN chmod +x /opt/iris-mpc-db-exporter

# nobody user
USER 65534

ENTRYPOINT ["/opt/iris-mpc-db-exporter"]
