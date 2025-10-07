## Emotion-based Music Recommendation System

A Streamlit application that detects a user's facial emotion from a webcam feed using a CNN and recommends songs that match the detected mood. Recommendations are sourced from a preprocessed music metadata file.

### Features
- Real-time face detection via OpenCV and `haarcascade_frontalface_default.xml`.
- CNN-based emotion recognition using TensorFlow/Keras.
- Curated music recommendations by dominant emotion.
- Streamlit UI with clickable links to song pages.

## Project Structure
```
Emotion-based-music-recommendation-system/
├─ app.py                          # Streamlit app entry point
├─ haarcascade_frontalface_default.xml
├─ model.h5                        # Pretrained CNN weights
├─ muse_v3.csv                     # Music metadata with emotion/valence tags
└─ README.md
```

## Prerequisites
- Python 3.9–3.11
- Webcam access (for emotion scanning)
- pip (or conda) for dependency installation

### Suggested Python packages
If you don't already have a `requirements.txt`, install these packages:

```bash
pip install streamlit opencv-python-headless opencv-python numpy pandas tensorflow==2.*
```

Notes:
- On Windows, if `opencv-python-headless` causes issues with camera display, install `opencv-python` instead.
- If you have a compatible GPU, you can install `tensorflow-gpu==2.*` (or the appropriate `tensorflow` extra) instead of CPU TensorFlow.

## Setup
1. Clone/download this repository.
2. Ensure the following files exist in the project root:
   - `model.h5`
   - `muse_v3.csv`
   - `haarcascade_frontalface_default.xml`
3. Verify Python dependencies are installed (see above).

## Important: Fix hardcoded paths
The current `app.py` references absolute Windows paths for `muse_v3.csv` and `model.h5`. To run from this repository directory, change those to relative paths.

In `app.py`, update the following lines:

```python
# BEFORE (example of absolute path)
df = pd.read_csv("C:\\Users\\User\\Desktop\\PRP\\FullyWorking\\Emotion-based-music-recommendation-system\\muse_v3.csv")
model.load_weights('C:\\Users\\User\\Desktop\\PRP\\FullyWorking\\Emotion-based-music-recommendation-system\\model.h5')

# AFTER (relative paths, assuming files are in project root)
df = pd.read_csv("muse_v3.csv")
model.load_weights("model.h5")
```

Also ensure the Haar Cascade is loaded correctly. The app currently uses OpenCV's built-in path:

```python
face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```

Alternatively, to load from the local file in the repo root:

```python
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```

## Run
From the project directory, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

### Quickstart (Windows PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; streamlit run app.py
```

### One-click start (Windows)
- Double-click `run.bat` in the project folder, or run:
```powershell
./run.bat
```

### Usage
1. Click "SCAN EMOTION (Click here)".
2. Look at your webcam; the app will sample a short sequence of frames.
3. After scanning, detected emotions are displayed and recommendations are listed with clickable links.

## Troubleshooting
- Camera not opening or black screen:
  - Close other apps using the webcam.
  - On Windows, grant Camera permissions to your terminal/IDE and browsers.
- TensorFlow install issues:
  - Use a Python version supported by your TensorFlow release (2.x typically supports 3.9–3.11).
  - Consider using a virtual environment: `python -m venv .venv && .venv\\Scripts\\activate` on Windows.
- OpenCV errors with highgui on servers/CI:
  - Use `opencv-python-headless` if you do not need GUI windows. The Streamlit app renders frames in the browser.
- Missing files:
  - Ensure `model.h5` and `muse_v3.csv` are present in the project root.

## Dataset/Attribution
- `muse_v3.csv` appears to contain music metadata and tags (e.g., Last.fm URLs, emotion/valence tags). Ensure you have the right to use and distribute this file. If it originates from an external dataset (e.g., Last.fm or a research corpus), please add the proper citation and/or link here.

## Security and Privacy
- The app uses your local webcam feed for emotion detection; frames are processed locally and not uploaded by default.
- Review the code before deploying publicly.

## License
No license has been specified. Consider adding a license (e.g., MIT) if you plan to share or collaborate publicly.

## Acknowledgements
- OpenCV for face detection.
- TensorFlow/Keras for the CNN.
- Streamlit for rapid UI development.

