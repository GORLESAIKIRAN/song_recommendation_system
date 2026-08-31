import unittest
from unittest.mock import patch
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
            app.CANONICAL_EMOTION_LABELS,
            (
                "Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad",
                "Surprise",
            ),
        )

    def test_ferplus_returns_only_supported_emotions(self):
        self.assertNotIn("Contempt", app.FERPLUS_LABELS)
        self.assertEqual(len(app.FERPLUS_LABELS), 8)

    def test_emotion_prediction_uses_trained_keras_model(self):
        class FakeModel:
            output_shape = (None, 7)

            def predict(self, values, verbose=0):
                self.values = values
                result = np.zeros((1, 7), dtype=np.float32)
                result[0, 3] = 1.0
                return result

        fake_model = FakeModel()
        original = app._emotion_model
        app._emotion_model = fake_model
        try:
            emotion, confidence = app._predict_emotion(
                np.full((80, 80), 255, dtype=np.uint8)
            )
        finally:
            app._emotion_model = original
        self.assertEqual(emotion, "Happy")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(float(fake_model.values.max()), 255.0)

    def test_health_reports_spotify_configuration(self):
        with app.app.test_client() as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("spotify_configured", response.get_json())
        self.assertIn("tensorflow_available", response.get_json())
        self.assertIn("onnx_emotion_available", response.get_json())

    def test_keyword_search_passes_the_exact_user_query(self):
        with patch.object(
            app, "_spotify_tracks", return_value=[]
        ) as search_tracks:
            with app.app.test_client() as client:
                response = client.post(
                    "/recommend_song",
                    json={"keyword": "Prabhas", "language": "Telugu"},
                )
        self.assertEqual(response.status_code, 404)
        search_tracks.assert_called_once_with("Prabhas Telugu songs")


if __name__ == "__main__":
    unittest.main()
