# Jupyter Notebook Setup Guide

This guide explains how to run Jupyter Notebook and access it either:
- **Over SSH** (secure remote access)
- **On local network** (access from other computers on the same network)

## Quick Start

### Option 1: Access Over Local Network

1. **On the server machine**, run:
   ```bash
   source venv/bin/activate
   ./start_jupyter_local.sh
   ```

2. **Note the server IP address** shown in the output (e.g., `192.168.1.100`)

3. **On any computer on the same local network**, open your browser and go to:
   ```
   http://SERVER_IP:8888
   ```
   (Replace `SERVER_IP` with the IP address from step 2)

4. **Enter the token** shown in the terminal output from step 1.

### Option 2: Access Over SSH (More Secure)

1. **On the remote server**, run:
   ```bash
   source venv/bin/activate
   ./start_jupyter_ssh.sh
   ```

2. **On your local machine**, set up SSH port forwarding:
   ```bash
   ssh -L 8888:localhost:8888 user@remote-server
   ```
   (Replace `user@remote-server` with your actual SSH connection details)

3. **Open your browser** and go to:
   ```
   http://localhost:8888
   ```

4. **Enter the token** shown in the terminal output from step 1.

## Detailed Setup

### Local Network Access

#### Method 1: Using the Helper Script (Recommended)

The `start_jupyter_local.sh` script automates the setup:

```bash
source venv/bin/activate
./start_jupyter_local.sh
```

This will:
- Generate a Jupyter config if needed
- Start Jupyter with network access enabled
- Display the server IP address and access URLs
- Show the token for authentication

#### Method 2: Manual Setup for Local Network

1. **Generate Jupyter config** (if not exists):
   ```bash
   source venv/bin/activate
   jupyter notebook --generate-config
   ```

2. **Set a password** (highly recommended for network access):
   ```bash
   jupyter notebook password
   ```

3. **Find your server's IP address**:
   ```bash
   hostname -I
   # or
   ip addr show
   ```

4. **Start Jupyter with network access**:
   ```bash
   jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
   ```

5. **Access from other computers**:
   - Open browser on any computer on the same network
   - Go to: `http://SERVER_IP:8888`
   - Enter the token or password

### SSH Access

#### Method 1: Using the Helper Script (Recommended)

The `start_jupyter_ssh.sh` script automates the setup:

```bash
source venv/bin/activate
./start_jupyter_ssh.sh
```

This will:
- Generate a Jupyter config if needed
- Start Jupyter with remote access enabled
- Display the URL and token

#### Method 2: Manual Setup for SSH

1. **Generate Jupyter config** (if not exists):
   ```bash
   source venv/bin/activate
   jupyter notebook --generate-config
   ```

2. **Set a password** (optional but recommended):
   ```bash
   jupyter notebook password
   ```

3. **Start Jupyter with remote access**:
   ```bash
   jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
   ```

4. **Note the token** from the output (looks like `?token=abc123...`)

### SSH Port Forwarding

**On your local machine**, connect with port forwarding:

```bash
ssh -L 8888:localhost:8888 user@remote-server
```

Or if you're already connected via SSH, you can set up port forwarding in a new terminal:

```bash
ssh -L 8888:localhost:8888 -N user@remote-server
```

The `-N` flag means "don't execute a remote command" - useful if you just want the port forwarding.

### Accessing Jupyter

Once port forwarding is active, open your browser and go to:
- `http://localhost:8888`
- Enter the token from the Jupyter output, or use the password you set

### Security Notes

**For Local Network Access:**
- ⚠️ **Less secure** - anyone on your local network can potentially access the server
- **Always set a password**: `jupyter notebook password` (highly recommended!)
- The token in the URL provides authentication - keep it secret
- Consider using a firewall to restrict access to specific IPs
- Only use on trusted networks (home/office), not public WiFi

**For SSH Access:**
- ✅ **More secure** - traffic is encrypted through SSH
- The token in the URL provides authentication - keep it secret
- Port forwarding (`-L`) is secure - traffic is encrypted through SSH
- Consider setting a password for additional security: `jupyter notebook password`

**General:**
- You can also use `jupyter lab` instead of `jupyter notebook` for JupyterLab
- Consider using HTTPS for production deployments

### Stopping Jupyter

Press `Ctrl+C` in the terminal where Jupyter is running, or find the process:
```bash
jupyter notebook list  # Shows running servers
jupyter notebook stop  # Stops the default server
```

### Troubleshooting

**Port already in use:**
- Use a different port: `jupyter notebook --port=8889`
- Update SSH forwarding: `ssh -L 8889:localhost:8889 ...`
- Or use the different port in the URL: `http://SERVER_IP:8889`

**Connection refused (Local Network):**
- Make sure Jupyter is running with `--ip=0.0.0.0`
- Check firewall settings on the server:
  ```bash
  # Ubuntu/Debian - allow port through firewall
  sudo ufw allow 8888/tcp
  # Or for specific IP:
  sudo ufw allow from 192.168.1.0/24 to any port 8888
  ```
- Verify both computers are on the same network
- Check if the server IP address is correct: `hostname -I`

**Connection refused (SSH):**
- Make sure Jupyter is running with `--ip=0.0.0.0`
- Check firewall settings on the remote server
- Verify SSH port forwarding is active: `ssh -L 8888:localhost:8888 ...`

**Can't find server IP address:**
- Try: `hostname -I`
- Or: `ip addr show | grep inet`
- Or: `ifconfig | grep inet`

**Token not working:**
- Copy the full token from the terminal output
- Or reset password: `jupyter notebook password`
- Make sure you're using the correct URL format: `http://IP:PORT/?token=TOKEN`

