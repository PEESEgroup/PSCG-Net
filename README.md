# PSCG-Net: A Multi-Scale Crystal Graph Neural Network for Accelerated Materials Discovery

> **Abstract:** Accelerating the discovery of materials with tailored properties is essential for advancements in energy, electronics, and sustainable technologies. While machine learning methods—especially graph neural networks (GNNs)—have shown promise in predicting material properties, existing models often fail to capture the long-range interactions inherent in crystalline materials due to fixed cutoff radii, limiting their predictive accuracy. We introduce the Pair-Scalable Crystal Graph Neural Network (PSCG-Net), a framework that addresses this gap by incorporating multi-scale structural representations inspired by the pair distribution function (PDF). By constructing graphs with multiple cutoff distances, PSCG-Net effectively captures both short-range and long-range atomic interactions. Evaluated on 152,063 crystal structures, PSCG-Net consistently outperforms the baseline CGCNN model, achieving a mean absolute error of 0.065 eV in predicting a typical property of formation energy. It demonstrates robust performance across six diverse datasets, underscoring its versatility. First-principles calculations using the hybrid functional confirm its superior accuracy in predicting band gap types. To showcase its practical utility, we used PSCG-Net to screen for high-performance materials in photovoltaics, dielectrics, and superconductors. By efficiently capturing hierarchical atomic interactions, PSCG-Net accelerates the design and discovery of novel materials and offers a generalizable framework that can be adapted to address intricate multi-scale challenges in diverse scientific fields, thereby providing a versatile asset for advancing research and innovation across disciplines.


## 🛠️ Environment Setup

Ensure your computing environment is properly configured by following these steps:

1. **Update Conda:**

    ```bash
    conda upgrade conda
    ```

2. **Create and Activate the PSCG-Net Environment:**

    ```bash
    conda create -n pscgnet python=3 scikit-learn pytorch torchvision pymatgen -c pytorch -c conda-forge
    conda activate pscgnet
    ```

## 📂 Dataset Preparation

The PSCG-Net model requires a specifically structured dataset for both training and prediction.

### 🔍 Requirements

- **Crystallographic Information Files ([CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File))**: These files should accurately describe the crystal structures of the materials of interest.
- **Target Properties**: Essential for training the model to map properties to structures. For prediction tasks, placeholder values are used in the dataset.

### 🗂️ Creating a Custom Dataset

Organize your dataset into a root directory (`root_dir`) with the following structure:

1. **`id_prop.csv`**: A [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file containing two columns:
    - **ID**: A unique identifier for each crystal.
    - **Property**: The target property value. For prediction tasks, you can use placeholder numbers.

    *Example:*

    | ID   | Property |
    |------|----------|
    | id0  | 1.23     |
    | id1  | 4.56     |
    | ...  | ...      |

2. **`cif/` Directory**: Contains `.cif` files corresponding to each unique ID.

    - **Example Files:**
        - `id0.cif`
        - `id1.cif`
        - ...

## 🏋️‍ Training the Model

To train a new PSCG-Net model, follow these steps:

1. **Prepare Your Dataset:**

    Ensure your `root_dir` is organized as described in the [Dataset Preparation](#-dataset-preparation) section.

2. **Initiate Training:**

    Navigate to the `cgcnn` directory and run:

    ```bash
    python main.py root_dir
    ```

3. **Customize Dataset Splits:**

    Adjust the training, validation, and testing ratios as needed:

    ```bash
    python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-regression
    ```

4. **For Classification Tasks:**

    Specify the task type during training:

    ```bash
    python main.py --task classification --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-classification
    ```

### 📄 Training Outputs

After training, the `runs` directory will contain:

- **`model_best.pth.tar`**: The best-performing model based on validation accuracy.
- **`checkpoint.pth.tar`**: A snapshot of the model at the end of the last epoch.
- **`test_results.csv`**: Results for each crystal in the test set, including IDs, target values, and predicted values.

## 🔮 Making Predictions

To make predictions using a pre-trained model:

1. **Prepare the Prediction Dataset:**

    Ensure your `root_dir` contains the necessary `.cif` files and a corresponding `id_prop.csv` with placeholder values.

2. **Run the Prediction Script:**

    ```bash
    python predict.py pre-trained.pth.tar root_dir
    ```

    - **For Classification Tasks:** The `test_results.csv` will include probabilities indicating the likelihood of each crystal belonging to specific classes.

## 📄 Results

After completing training and predictions, refer to the `runs` directory for detailed outcomes, including model performance metrics and prediction results.

## Citation
If you find our work and this repository useful, please consider giving a star :star: and citation :beer::

```
@misc{chen2024future,
      title={PSCG-Net: A Multi-Scale Crystal Graph Neural Network for Accelerated Materials Discovery}, 
      author={Guangyao Chen and Zhilong Wang and Fengqi You},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
