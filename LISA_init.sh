module purge
module load pre2019
python3 --version
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

echo "Installing requirements.txt..."
pip3 install --user -r requirements.txt
echo "Installing other packages..."
pip3 install --user transformers==2.3.0 torch==1.2.0 tqdm==4.48.2
pip3 install --user faiss-gpu
pip3 install --user pytrec_eval
