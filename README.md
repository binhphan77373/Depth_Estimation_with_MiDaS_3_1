Based on the gathered information, here is a draft for your README:

# Depth Estimation with MiDaS 3.1

This repository provides an implementation of depth estimation using MiDaS 3.1 models.

## Introduction

This project uses MiDaS, a state-of-the-art model for monocular depth estimation. The implementation supports three different model types with varying levels of accuracy and inference speed.

## Model Types

- **DPT_LARGE**: Highest accuracy, slowest inference speed.
- **DPT_HYBRID**: Medium accuracy, medium inference speed.
- **MIDAS_SMALL**: Lowest accuracy, highest inference speed.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/binhphan77373/Depth_Estimation_with_MiDaS_3_1.git
    cd Depth_Estimation_with_MiDaS_3_1
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Model

To run the model with the default settings, execute:
```sh
python model.py
```

This will use the `MIDAS_SMALL` model type by default. You can change the model type by modifying the last line in `main.py`:
```python
if __name__ == '__main__':
    run(ModelType.DPT_LARGE)
```

### Live Prediction

The script supports live depth prediction using a webcam. Simply run the script and press 'q' to quit:
```sh
python model.py
```

## Example

The script reads frames from your webcam, predicts the depth map, and displays the original and depth map side by side.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MiDaS model from the [Intel ISL GitHub](https://github.com/isl-org/MiDaS).
