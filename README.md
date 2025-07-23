<h3>Handwritten Digit Recognition (From Scratch)</h3>
<p>A self-contained handwritten digit recognition system implemented using **NumPy** — no TensorFlow, PyTorch, or scikit-learn.</p>

---
<h3>Repository Structure</h3>
<pre>
Handwritten-digit-recognition/
├── basicWith_NN            # Fodler containing a jupyter notebook file which explain the whole implementation.
├── With_raw_image          # Folder containing 
    ├── RAW_image           # Folder containing RAW images
    ├── ProcessedImage      # Folder containing preprocessed training and testing immages
    ├── preprocess.py       # File containing the preprocessing program
    ├── mainNN.py           # File containing the main NN program
├── model_application       # Folder containing an application program which is baesd on mainNN.py output data
</pre>

<h3>Requirements<h3>
<pre>
- Python 3.8+
- NumPy
- OpenCV (for reading image files)
</pre>

<h3>Install dependencies via:</h3>
<pre>pip install numpy opencv-python</pre>

<h3>Dataset Format</h3>
<pre>
- **train/**: 28×28 grayscale images.
- **test/**: Similar format for evaluation.
</pre>


<h3>Model Overview</h3>
<pre>
- **Input:** 784 neurons (flattened 28×28 image)
- **Output Layer:** 10 neurons (digits 0–9)
- **Activation:** Sigmoid
- **Loss:** Mean Squared Error (MSE)
- **Training:** Gradient Descent
</pre>

<h3>Weights Storage</h3>
<pre>Model parameters (weights & biases) are saved using `numpy.savez()` into the `weights/` folder after training. These are automatically reloaded during testing.</pre>

<h3>Steps<h3>
<pre>
1. "basicWith_NN" folder
2. "With_raw_image" folder
3. "model_application" folder
</pre>

<h3>Author</h3>
**Tanveer Islam** — [GitHub @Tanveer‑2002](https://github.com/Tanveer-2002)

Feel free to give a ⭐ if you found this helpful!

---
