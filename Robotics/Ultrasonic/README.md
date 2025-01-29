# Ultrasonic Distance Measurement using Raspberry Pi
This script measures distance using an **HC-SR04 Ultrasonic Sensor** connected to a Raspberry Pi. It calculates the time taken for the ultrasonic pulse to bounce back and converts it into distance.  

## Requirements 
- Raspberry Pi with Raspbian OS  
- HC-SR04 Ultrasonic Sensor  
- Jumper wires 

## How It Works
#### 1. Imports Necessary Libraries
- RPi.GPIO for GPIO control
- time for handling delays

#### 2. Configures GPIO Pins
- Uses BCM mode for GPIO numbering
- TRIG (GPIO 23): Sends ultrasonic pulse
- ECHO (GPIO 24): Receives the reflected pulse

#### 3. Sends Ultrasonic Pulse
- Triggers a 10-microsecond pulse on TRIG
- Measures the time taken for the pulse to return

#### 4. Calculates Distance
- Uses the formula:

        Distance = Time × 17150
  
- Rounds the result to 2 decimal places
- Displays the measured distance in cm

#### 5. Cleans Up GPIO
- Ensures safe GPIO handling after execution

## Circuit Connections

| HC-SR04 Pin | Raspberry Pi Pin |
|-------------|------------------|
| VCC         | 5V               |
| GND         | GND              |
| TRIG        | GPIO 23          |
| ECHO        | GPIO 24          |

## Expected Output
- The script will print the measured distance in centimeters.
- Example output:

      Distance Measurement In Progress  
      Waiting For Sensor To Settle
      Distance: 15.24 cm  

## Notes
- Ensure correct wiring for accurate results.
- If the sensor is unresponsive, increase the sleep time before sending the pulse.
- Use a 5V to 3.3V voltage divider on the ECHO pin to prevent damaging the Raspberry Pi.

