#!/bin/bash
# Deployment script for OpenWeather

set -e

# Configuration
APP_NAME="openweather"
APP_DIR="/opt/openweather"
VENV_DIR="$APP_DIR/venv"
SYSTEMD_SERVICE_NAME="openweather-api"
SYSTEMD_SERVICE_FILE="/etc/systemd/system/$SYSTEMD_SERVICE_NAME.service"
LOG_DIR="/var/log/openweather"
USER="openweather"
GROUP="openweather"

echo "=== OpenWeather Deployment Script ==="

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root"
    exit 1
fi

# Create user if it doesn't exist
if ! id -u $USER &>/dev/null; then
    echo "Creating user $USER..."
    useradd -m -s /bin/bash $USER
fi

# Create application directory
echo "Creating application directory..."
mkdir -p $APP_DIR
mkdir -p $LOG_DIR

# Copy application files
echo "Copying application files..."
rsync -av --exclude="venv" --exclude=".git" --exclude="__pycache__" . $APP_DIR/

# Setup virtual environment
echo "Setting up virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -e $APP_DIR

# Copy environment file if not exists
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Copying .env.example to .env..."
    cp $APP_DIR/.env.example $APP_DIR/.env
    echo "IMPORTANT: Please edit $APP_DIR/.env with your API keys and configuration"
fi

# Create systemd service
cat > $SYSTEMD_SERVICE_FILE << EOF
[Unit]
Description=OpenWeather API Service
After=network.target

[Service]
User=$USER
Group=$GROUP
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/python -m openweather.master api
Restart=on-failure
StandardOutput=append:$LOG_DIR/api.log
StandardError=append:$LOG_DIR/api-error.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
echo "Setting permissions..."
chown -R $USER:$GROUP $APP_DIR
chown -R $USER:$GROUP $LOG_DIR
chmod 644 $SYSTEMD_SERVICE_FILE

# Enable and start service
echo "Enabling and starting service..."
systemctl daemon-reload
systemctl enable $SYSTEMD_SERVICE_NAME
systemctl restart $SYSTEMD_SERVICE_NAME

echo "=== Deployment Complete ==="
echo "API service running at http://localhost:8000"
echo "Logs available at $LOG_DIR"
echo "Service status: systemctl status $SYSTEMD_SERVICE_NAME"