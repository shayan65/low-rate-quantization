#!/bin/bash
ssh -t -o ConnectTimeout=10 shayan@192.168.1.26 'bash /home/shayan/install-tailscale-workstation.sh'
setup_result=$?
echo "Workstation setup exit code: $setup_result"
read -r -p 'Press Enter to close this window. '
