# **WMT25 Submission Quick Guide**

## **Overview**
WMT25 Model Compression requires a **Docker image** containing your compressed model with a specific interface for evaluation.

## **Simple 4-Step Process**

### **Step 1: Prepare Your Compressed Model**
Ensure you have a compressed model directory with:
- Model files (e.g., `pytorch_model.bin`, `config.json`)
- Tokenizer files (e.g., `tokenizer.json`, `tokenizer_config.json`)
- Any other files needed for inference

### **Step 2: Create Your run.sh Script**
Your `run.sh` script MUST follow this exact interface:

```bash
#!/usr/bin/env bash
set -eu

# Required WMT25 interface: run.sh $lang_pair $batch_size
langs=$1
batch_size=$2

# Get the directory where this script is located (your model directory)
mydir=$(dirname "$0")
mydir=$(realpath "$mydir")

# Your translation command here - must read from stdin and write to stdout
# Example using the framework's baseline.py:
python -m modelzip.baseline $langs $batch_size -m $mydir

# Alternative: if you have a custom run.py script in your model directory:
# python $mydir/run.py $langs $batch_size
```

**Key Requirements for run.sh:**
- **Arguments**: Must accept exactly 2 arguments: `$lang_pair` (e.g., `ces-deu`) and `$batch_size` (integer)
- **Input/Output**: Must read from stdin and write to stdout
- **Working Directory**: Must work from any directory (use `$mydir` for model files)
- **Executable**: Must be executable (`chmod +x run.sh`)
- **Offline**: Must work without internet access

**Different Ways to Create run.sh:**

1. **Using modelzip framework** (if you used this framework for compression):
```bash
#!/usr/bin/env bash
set -eu
langs=$1
batch_size=$2
mydir=$(dirname "$0")
mydir=$(realpath "$mydir")
python -m modelzip.baseline $langs $batch_size -m $mydir
```

2. **Using custom Python script** (create your own run.py):
```bash
#!/usr/bin/env bash
set -eu
langs=$1
batch_size=$2
mydir=$(dirname "$0")
mydir=$(realpath "$mydir")
python $mydir/run.py $langs $batch_size
```

3. **Direct Python in run.sh** (inline Python code):
```bash
#!/usr/bin/env bash
set -eu
langs=$1
batch_size=$2
mydir=$(dirname "$0")
mydir=$(realpath "$mydir")

python -c "
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model from current directory
model = AutoModelForCausalLM.from_pretrained('$mydir', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('$mydir')

# Your translation logic here
# Read from stdin, process, write to stdout
"
```

### **Step 3: Prepare Model Directory for Submission**
```bash
# Create submission directory
mkdir -p /tmp/submission_prep/your-model-name

# Copy your model files
cp -r /path/to/your/compressed/model/* /tmp/submission_prep/your-model-name/

# Create or copy your run.sh script
cp /path/to/your/run.sh /tmp/submission_prep/your-model-name/
chmod +x /tmp/submission_prep/your-model-name/run.sh

# Test your run.sh script locally
echo "Test sentence" | /tmp/submission_prep/your-model-name/run.sh ces-deu 1
```

### **Step 4: Create Docker Image**
Create a `Dockerfile` (replace `YourTeam` with your team name):
```dockerfile
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && apt install -y python3 python3-pip git wget curl && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch for CUDA 12.6
RUN pip install --no-cache-dir torch torchvision transformers accelerate

# Optional: Install additional dependencies if using the modelzip framework
# WORKDIR /work/wmt25-model-compression
# COPY requirements.txt pyproject.toml ./
# COPY modelzip/ modelzip/
# RUN pip install --no-cache-dir -e ./

# Copy your model to required WMT25 location
COPY /tmp/submission_prep/your-model-name /model/your-submission-id
RUN chmod +x /model/your-submission-id/run.sh
```

### **Step 5: Build and Submit**
```bash
# Build Docker image
docker build -t YourTeam-dockerimage .

# Test it works
echo "Test sentence" | docker run -i YourTeam-dockerimage /model/your-submission-id/run.sh ces-deu 1

# Create submission package
docker save --output YourTeam-dockerimage.tar YourTeam-dockerimage
sha512sum YourTeam-dockerimage.tar

# Upload tar file to public repository (Google Drive, etc.)
# Submit the download link + SHA512 to WMT25 form
```

## **Key Requirements**
- **Interface**: `run.sh $lang_pair $batch_size` (reads stdin, writes stdout)
- **Location**: Model at `/model/your_submission_id/`
- **Naming**: `YourTeam-dockerimage.tar` (no spaces in team name)
- **Offline**: Must work without internet access
- **Testing**: Verify with `ces-deu`, `jpn-zho`, `eng-ara` language pairs

## **Quick Validation**
```bash
# Test all language pairs
for lang in ces-deu jpn-zho eng-ara; do
    echo "Testing $lang..." | docker run -i YourTeam-dockerimage /model/your_submission_id/run.sh $lang 1
done

# Test offline (no internet)
docker run --network none -i YourTeam-dockerimage /model/your_submission_id/run.sh ces-deu 1 <<< "Offline test"
```

## **Complete Example Workflow**

Here's a complete example for submitting a compressed model:

```bash
# 1. Prepare your compressed model directory
# Assume you have a compressed model at: /path/to/my/compressed/model/
# It should contain: pytorch_model.bin, config.json, tokenizer files, etc.

# 2. Create run.sh script
mkdir -p /tmp/submission_prep/my-compressed-model
cp -r /path/to/my/compressed/model/* /tmp/submission_prep/my-compressed-model/

# Create the run.sh script
cat > /tmp/submission_prep/my-compressed-model/run.sh << 'EOF'
#!/usr/bin/env bash
set -eu

# Required WMT25 interface
langs=$1
batch_size=$2

# Get model directory
mydir=$(dirname "$0")
mydir=$(realpath "$mydir")

# Run translation (using transformers library directly)
python -c "
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('$mydir', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('$mydir')

# Read input
lines = [line.strip() for line in sys.stdin.readlines()]

# Simple translation logic (customize based on your model)
for line in lines:
    # Create translation prompt
    prompt = f'Translate from {langs.split(\"-\")[0]} to {langs.split(\"-\")[1]}: {line}'
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    # Decode and print result
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(result.strip())
"
EOF

chmod +x /tmp/submission_prep/my-compressed-model/run.sh

# 3. Test your run.sh script locally
echo "Hello world" | /tmp/submission_prep/my-compressed-model/run.sh ces-deu 1

# 4. Create Dockerfile with team name (example: MyTeam)
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && apt install -y python3 python3-pip git wget curl && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision transformers accelerate

# Copy your model to required location
COPY /tmp/submission_prep/my-compressed-model /model/my-compressed-model
RUN chmod +x /model/my-compressed-model/run.sh
EOF

# 5. Build and test Docker
docker build -t MyTeam-dockerimage .
echo "Test translation" | docker run -i MyTeam-dockerimage /model/my-compressed-model/run.sh ces-deu 1

# 6. Create submission package
docker save --output MyTeam-dockerimage.tar MyTeam-dockerimage
sha512sum MyTeam-dockerimage.tar

# 7. Upload to public repository and submit to WMT25
```

## **🛠️ Troubleshooting**

### **Common Issues**

1. **"Model not found" error**
   - Ensure your model files are copied to the correct Docker location
   - Check that `run.sh` is executable: `chmod +x run.sh`

2. **"No such file or directory" error**
   - Verify the model path in your Dockerfile COPY command matches your actual model location

3. **Docker build fails**
   - Ensure all required files exist before building
   - Check that `requirements.txt` and `pyproject.toml` are in the project root

4. **Translation produces no output**
   - Test with a simple sentence first
   - Check Docker logs: `docker logs <container_id>`

### **Quick Debugging**
```bash
# Enter Docker container for debugging
docker run -it YourTeam-dockerimage bash

# Check model files exist
ls -la /model/your_submission_id/

# Test components individually
python -m modelzip.baseline --help
```

## **Resources**

- **Framework Overview**: Check `README.md` for basic usage of the modelzip framework
- **WMT25 Official Page**: https://www2.statmt.org/wmt25/model-compression.html