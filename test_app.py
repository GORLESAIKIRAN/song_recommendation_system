import unittest
import numpy as np

import app


class CelebrityRecognitionTests(unittest.TestCase):
    def test_face_candidate_extraction_returns_at_least_one_region(self):
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        regions = app._extract_face_candidates_from_array(image)
        self.assertGreaterEqual(len(regions), 1)

    def test_label_lookup_builds_index_map(self):
        lookup = app._build_label_lookup({"Akhil": 0, "Nani": 1})
        self.assertEqual(lookup[0], "Akhil")
        self.assertEqual(lookup[1], "Nani")

    def test_emotion_labels_match_model_output_order(self):
        self.assertEqual(
            app.EMOTION_LABELS,
            (
                "Angry", "Disgusted", "Fear", "Happy", "Neutral", "Sad",
                "Surprise",
            ),
        )

    def test_health_reports_spotify_configuration(self):
        with app.app.test_client() as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("spotify_configured", response.get_json())


if __name__ == "__main__":
    unittest.main()
