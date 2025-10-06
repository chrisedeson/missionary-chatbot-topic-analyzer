#!/usr/bin/env bash
# exit on error
set -o errexit

# Configure Cargo for Render's read-only file system
export CARGO_HOME=/tmp/cargo
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
export CARGO_NET_GIT_FETCH_WITH_CLI=true
mkdir -p $CARGO_HOME

# Create a minimal cargo config to ensure it uses the right paths
mkdir -p $CARGO_HOME
cat > $CARGO_HOME/config.toml << EOF
[registries.crates-io]
protocol = "sparse"

[net]
git-fetch-with-cli = true
EOF

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Generate Prisma client
npx prisma generate