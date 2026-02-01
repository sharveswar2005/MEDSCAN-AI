import { useState } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setResult(null);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleSubmit = async () => {
    if (!file) return alert("Upload X-ray image");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(
        "http://127.0.0.1:8000/predict-with-heatmap",
        formData
      );

      setResult(res.data);
    } catch (err) {
      alert("Prediction failed");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-6">

      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-xl p-7 space-y-5 animate-fadeIn">

        {/* Header */}
        <h1 className="text-3xl font-bold text-center text-slate-800">
          MedScan AI
        </h1>

        <p className="text-center text-gray-500 text-sm">
          AI-powered Pneumonia Detection System
        </p>

        {/* Upload Section */}
        <div className="border-2 border-dashed rounded-lg p-4 text-center hover:border-blue-500 transition">

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="w-full"
          />

          {preview && (
            <img
              src={preview}
              alt="preview"
              className="mx-auto mt-3 rounded shadow max-h-40"
            />
          )}
        </div>

        {/* Button */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 active:scale-95 transition transform text-white font-semibold py-2 rounded-lg"
        >
          {loading ? "Analyzing X-ray..." : "Analyze X-ray"}
        </button>

        {/* Loader */}
        {loading && (
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        )}

        {/* Result Section */}
        {result && (
          <div className="bg-gray-100 p-4 rounded-lg space-y-3 animate-slideUp">

            {/* Diagnosis */}
            <div className="flex justify-between items-center">
              <span className="font-semibold text-gray-700">
                Diagnosis
              </span>

              <span
                className={`px-4 py-1 rounded-full text-white text-sm font-semibold shadow ${
                  result.prediction === "PNEUMONIA"
                    ? "bg-red-600 animate-pulse"
                    : "bg-green-600"
                }`}
              >
                {result.prediction}
              </span>
            </div>

            {/* Confidence */}
            <div>
              <p className="text-sm font-medium text-gray-600">
                Confidence: {result.confidence.toFixed(2)}%
              </p>

              <div className="w-full bg-gray-300 rounded h-2 mt-1 overflow-hidden">
                <div
                  className="bg-blue-600 h-2 rounded transition-all duration-1000 shadow"
                  style={{
                    width: `${Math.min(result.confidence, 100)}%`
                  }}
                />
              </div>
            </div>

            {/* Heatmap */}
            {result.heatmap && (
              <div>
                <p className="text-sm font-semibold text-gray-700 mb-1">
                  Grad-CAM Visualization
                </p>

                <img
                  src={`data:image/jpeg;base64,${result.heatmap}`}
                  alt="GradCAM"
                  className="rounded-lg border shadow mx-auto"
                />
              </div>
            )}

          </div>
        )}

      </div>

    </div>
  );
}
