#!/bin/bash
# Setup Binance Testnet API keys for paper trading

# Clear screen and show header
clear
echo "=====================================
BINANCE TESTNET SETUP FOR PAPER TRADING
====================================="

# Check if keys are already set
if [[ -n "${BINANCE_API_KEY_TEST}" && -n "${BINANCE_API_SECRET_TEST}" ]]; then
  echo "✅ Testnet API keys are already configured in your environment."
  echo "Current API Key: ${BINANCE_API_KEY_TEST:0:4}...${BINANCE_API_KEY_TEST: -4}"
  echo ""
  echo "To update them, run this script again."
  
  # Ask if user wants to continue anyway
  read -p "Do you want to update your API keys? (y/n): " should_continue
  if [[ "$should_continue" != "y" ]]; then
    exit 0
  fi
fi

# Set the API key
TESTNET_API_KEY="Z2fyi98I55Df65Q6Erofsw6PtbOjZPKzJTtUYsXDhKDPPHHYyP2yz4t4U4KE0rwA"

# Prompt for the secret key (without displaying it)
echo ""
echo "Please enter your Binance Testnet Secret Key (input will be hidden):"
read -s TESTNET_API_SECRET
echo ""

# Display Binance Testnet instructions
echo "
To get your own Binance Testnet API keys:
1. Go to https://testnet.binancefuture.com/en/futures
2. Create a testnet account
3. Generate API keys from your testnet dashboard
"

if [[ -z "$TESTNET_API_SECRET" ]]; then
  echo "⚠️ ERROR: Secret key cannot be empty!"
  exit 1
fi

echo "Setting up environment variables for this terminal session..."
export BINANCE_API_KEY_TEST="$TESTNET_API_KEY"
export BINANCE_API_SECRET_TEST="$TESTNET_API_SECRET"

echo "✅ Environment variables set successfully for this terminal session!"
echo ""
echo "To make these permanent, add these lines to your ~/.zshrc or ~/.bash_profile:"
echo "export BINANCE_API_KEY_TEST=\"$TESTNET_API_KEY\""
echo "export BINANCE_API_SECRET_TEST=\"your_secret_key\""
echo ""
echo "To test your configuration, run:"
echo "python3 scripts/paper_trade.py --symbol BTCUSDT --interval 1d --trading-tier regular"
