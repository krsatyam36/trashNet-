# Hailo-8 YOLOv8 Compilation Pipeline (320x320, Headless)

End to end Python pipeline to convert a PyTorch YOLOv8n model into a **Hailo Executable Format (.hef)** binary optimized for the **Hailo-8 AI accelerator**.
Designed specifically to overcome routing constraints on edge hardware while maintaining reliable detection performance.

---

## Model & Dataset Sources

This pipeline uses pre-trained, off-the-shelf resources:

* **Model**:
  Bluelabel Satellite Equipment Detection YOLOv8n (VHR10)
  [https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10](https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10)

* **Calibration Dataset**:
  VHR-10 Satellite Image Dataset
  [https://huggingface.co/datasets/satellite-image-deep-learning/VHR-10](https://huggingface.co/datasets/satellite-image-deep-learning/VHR-10)

---

## Architecture Decisions & Hardware Optimizations

To resolve Hailo routing limitations such as `Agent Infeasible` errors at `concat14`, the pipeline applies three key optimizations:

### 1️⃣ Resolution Downscaling

* Input resolution is fixed to **320x320**
* Reduces memory footprint and interconnect bandwidth
* Enables successful compiler mapping on Hailo-8

### 2️⃣ Headless Parsing

* YOLOv8 detection head is explicitly removed during ONNX parsing
* The Hailo chip runs only the convolutional backbone and neck
* Bounding box decoding and NMS are executed on the host CPU
* Prevents routing bottlenecks during compilation

### 3️⃣ Hardware-Level Normalization

* A `0–255 → 0–1` normalization layer is injected directly into the hardware graph
* Offloads preprocessing from the host CPU
* Ensures consistent input scaling on-device

---

# Execution Workflow

Run scripts sequentially from the project root or `scripts/` directory.
Ensure your **Hailo DFC virtual environment** is activated before starting.

---

## Step 1: Export PyTorch → ONNX

```bash
python3 step1-export-onnx-320x320.py
```

**Purpose**

* Converts the `.pt` model into `.onnx`
* Enforces strict `320x320` input shape
* Keeps memory usage small enough for successful Hailo mapping

**Output**

* `model_320.onnx`

---

## Step 2: Parse ONNX → Headless HAR

```bash
python3 step2-parse-headless-320x320.py
```

**Purpose**

* Converts `.onnx` to Hailo Archive (`.har`)
* Slices graph at the 6 convolutional outputs before detection head
* Isolates input node
* Removes YOLO detection head to avoid compiler crashes

**Output**

* `model_headless.har`

---

## Step 2.5: Generate Calibration Dataset

```bash
cd dataset/
python3 create-dataset-320x320.py
cd ..
```

**Purpose**

* Downloads a small subset (64 images) of VHR-10
* Resizes to `320x320`
* Applies YOLO preprocessing:

  * RGB conversion
  * Channel-first formatting
  * 0.0–1.0 scaling
* Saves 4D NumPy tensor for quantization

**Output**

* `calib_data_320.npy`

---

## Step 3: Resize & Quantize (FP32 → INT8)

```bash
python3 step3-resize-quantize-320x320.py
```

**Purpose**

* Converts FP32 model to INT8
* Uses calibration data to track activation ranges
* Preserves detection accuracy under 8-bit compression
* Prepares model for efficient edge execution

**Output**

* `model_quantized.har`

---

## Step 4: Final Compilation (HAR → HEF)

```bash
python3 step4-har-hef-320x320.py
```

**Purpose**

* Maps quantized model onto Hailo-8 hardware architecture
* Injects hardware normalization layer
* Finalizes binary for deployment

**Output**

* `model_320.hef`

---

# Final Artifact

The final deliverable is:

```
model_320.hef
```

This binary:

* Runs on Hailo-8
* Executes backbone + neck on-chip
* Requires host-side post-processing for:

  * Bounding box decoding
  * Non-Maximum Suppression (NMS)

---



