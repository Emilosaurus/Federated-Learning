# Federated Learning Setup 
## Overview
This repository contains code for setting up a Federated Learning (FL) server and client setup using Flower (FLwr). The FL server orchestrates model training across multiple clients, each with its own dataset.

## Requirements
- Python 3.7 or higher
- TensorFlow (for server and potentially clients)
- Flower (`flwr` library)
- PIL (Python Imaging Library) for image preprocessing in clients

## Server Setup
1. **Install Dependencies:**
   ```bash
   pip install tensorflow flwr pillow
   ```

2. **Run the Server:**
   - Ensure you have the server script (`server.py`) configured with the desired parameters, such as number of rounds and strategy.
   - Start the server:
     ```bash
     python server.py
     ```
   - The server will initialize and wait for clients to connect.

## Client Setup
1. **Client Requirements:**
   - Each client needs to have Python installed along with the necessary libraries (`numpy`, `PIL`, etc.).

2. **Client Code:**
   - The client code (`client.py`) should be configured to connect to the server address and participate in training rounds.

3. **Dataset Preparation:**
   - Clients should organize their datasets locally. In this example, datasets are organized in folders where each folder corresponds to a client's data category.

4. **Dataset Loading:**
   - Adjust the client script to load data from local directories (`D:/Git Uploads/Federated-Learning/test/C1` in this case). Ensure each client's dataset is appropriately resized and normalized.

## Running the Federated Learning Process
1. **Start Clients:**
   - Run multiple instances of the client script (`python client.py`) on different machines or as different processes on the same machine.
   - Clients will connect to the server and participate in training rounds as configured in `server.py`.

2. **Monitoring:**
   - Monitor the server logs for updates on rounds, client connections, and training progress.
   - Optionally, implement logging or visualizations to track model performance and convergence across clients.

## Additional Notes
- Adjust `num_rounds`, `fraction_fit`, and other parameters in `server.py` to suit your experiment requirements.
- Ensure all dependencies are installed and paths are correctly configured in both server and client scripts.
