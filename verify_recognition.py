import sys
import numpy as np

sys.path.insert(0, r"C:/Users/MOHAN M/Downloads/song_recommendation")
import app

image = np.zeros((240, 240, 3), dtype=np.uint8)
print(len(app._extract_face_candidates_from_array(image)))
print(app._build_label_lookup({"Akhil": 0, "Nani": 1}))
