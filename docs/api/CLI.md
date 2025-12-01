# CLI Reference

## Commands

### `pipeline`

Main CLI command for pipeline operations.

```bash
pipeline [OPTIONS] COMMAND [ARGS]
```

**Options:**
- `-v, --verbose`: Enable verbose logging

**Commands:**
- `health`: Check pipeline health
- `server`: Start health check server

### `pipeline health`

Check pipeline health status.

```bash
pipeline health [OPTIONS]
```

**Options:**
- `--output-path PATH`: Output path for disk space check
- `-v, --verbose`: Enable verbose logging

**Example:**
```bash
pipeline health --output-path s3://bucket/output/
```

### `pipeline server`

Start health check HTTP server.

```bash
pipeline server [OPTIONS]
```

**Options:**
- `--port PORT`: Port to listen on (default: 8080)
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `-v, --verbose`: Enable verbose logging

**Example:**
```bash
pipeline server --port 8080 --host 0.0.0.0
```

## Environment Variables

- `RAY_ADDRESS`: Ray cluster address
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENABLE_HEALTH_SERVER`: Enable health server (true/false)
- `HEALTH_CHECK_PORT`: Health server port
- `OUTPUT_PATH`: Pipeline output path

