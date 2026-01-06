# tinyml-embedded-inference

# Embedded TinyML Inference Pipeline

This project demonstrates an embedded-style machine learning pipeline designed for microcontroller deployment.

The system includes:
- Offline training of a compact neural network
- Post-training INT8 quantization
- Export to a C header for microcontroller inference
- A simulated embedded inference demo measuring latency and output stability

Target platform: RP2040-class microcontrollers (e.g., Raspberry Pi Pico)

## Repository Structure

- training/  
  Training, quantization, and demo inference scripts.

- pico_firmware/  
  Generated model headers and example firmware integration.

## Demo

A video demonstration of the inference loop and runtime behavior accompanies this repository and is submitted separately via SlideRoom.

