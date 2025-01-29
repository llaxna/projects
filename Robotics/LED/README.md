# PWM LED Brightness Control using Raspberry Pi (RPi.GPIO)
This script controls an LED’s brightness using Pulse Width Modulation (PWM) on a Raspberry Pi. The LED gradually brightens and dims in a loop.

## Requirements
- Raspberry Pi with Raspbian OS
- Python 3 installed
- LED connected to GPIO 18 (PWM-capable pin)
- Resistor (e.g., 330Ω) in series with the LED

## How It Works
#### 1. Imports Necessary Libraries
- RPi.GPIO for GPIO control
- time.sleep for delay handling
  
#### 2. Configures GPIO Settings
- Uses **BCM mode** for GPIO numbering
- Sets **GPIO 18** as an output pin
- Disables warnings to avoid redundant messages

#### 3. Initializes PWM
- Configures **GPIO 18** for PWM at **1000 Hz**
- Starts PWM with an initial duty cycle of **0%**

#### 4. Brightness Control Loop
- Gradually increases brightness from **0% to 100%**
- Holds at maximum brightness for **0.5 seconds**
- Gradually decreases brightness back to **0%**
- Holds at minimum brightness for **0.5 seconds**
- Repeats the cycle indefinitely

## Circuit Diagram
Connect the LED as follows:
- **LED Anode (+) → GPIO 18** (via a resistor)
- **LED Cathode (-) → GND**

## Expected Output
- The LED will smoothly fade in and out continuously.
- The terminal will print the current **duty cycle** percentage.

## Notes
- Ensure your LED is connected to a **PWM-supported GPIO pin**.
- To stop the script, press **Ctrl + C**.
- If the LED does not respond, check your wiring and GPIO mode.
