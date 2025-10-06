#!/usr/bin/env bash
# exit on error
set -o errexit

# Set CARGO_HOME to a writable directory for Render's read-only file system
export CARGO_HOME=/tmp/cargo
mkdir -p $CARGO_HOME

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Generate Prisma client
npx prisma generate