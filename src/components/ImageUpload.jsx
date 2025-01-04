import React, { useState } from 'react';

const ImageUploadSection = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null); // State for audio playback
  const [audioResult, setAudioResult] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
    setProcessedImage(null);
    setResult(null);
    setError(null);
  };

  const handleAudioChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("audio/")) {
      setSelectedAudio(file);
      setAudioResult(null);
      setAudioError(null);
      // Create an object URL for the uploaded audio file
      setAudioPreview(URL.createObjectURL(file));
    } else {
      alert("Please upload a valid audio file.");
    }
  };

  const handleSubmit = async () => {
    if (selectedImage) {
      alert(`Image ${selectedImage.name} will be processed!`);
      setLoading(true);
      setError(null);

      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64String = reader.result.split(',')[1];

        try {
          const response = await fetch('http://127.0.0.1:5000/api/process-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64String }),
          });

          const result = await response.json();
          if (result.status === 'success') {
            setResult(result.results);
            setProcessedImage(result.image);
          } else {
            setError(result.message);
          }
        } catch (error) {
          console.error('Error processing image:', error);
          setError('There was an error processing the image.');
        } finally {
          setLoading(false);
        }
      };

      reader.readAsDataURL(selectedImage);
    } else {
      alert('Please select an image first.');
    }
  };

  const handleAudioSubmit = async () => {
    if (selectedAudio) {
      alert(`Audio ${selectedAudio.name} will be processed!`);
      setAudioLoading(true);
      setAudioError(null);

      const formData = new FormData();
      formData.append('audio', selectedAudio);

      try {
        const response = await fetch('http://127.0.0.1:5000/api/process-audio', {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();
        if (result.status === 'success') {
          setAudioResult(result.results);
        } else {
          setAudioError(result.message);
        }
      } catch (error) {
        console.error('Error processing audio:', error);
        setAudioError('There was an error processing the audio file.');
      } finally {
        setAudioLoading(false);
      }
    } else {
      alert('Please select an audio file first.');
    }
  };

  const formatDescription = (shape, description) => {
    return `${shape} - ${description.replace(/\n/g, ' ').trim()}`;
  };

  return (
    <section className="container mx-auto py-16" id="image-upload">
      <h2 className="text-3xl font-bold mb-6 text-center text-white">Upload Your Image for AI Processing</h2>
      <div className="flex flex-col items-center">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-4 p-2 border rounded text-white bg-gray-800"
        />
        <button
          onClick={handleSubmit}
          className="bg-blue-500 text-white py-2 px-6 rounded hover:bg-blue-600 transition"
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Process Image'}
        </button>
        {selectedImage && (
          <div className="mt-4">
            <h3 className="font-semibold text-center text-white text-lg">Selected Image Preview:</h3>
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Selected Preview"
              className="mt-2 max-w-xs rounded shadow-lg"
            />
          </div>
        )}
        {error && (
          <div className="mt-4 bg-red-500 text-white p-4 rounded">
            <p>Error: {error}</p>
          </div>
        )}
        {processedImage && (
          <div className="mt-8 flex flex-col items-center">
            <h3 className="text-center text-lg text-white mb-4">Processed Image:</h3>
            <img
              src={`data:image/png;base64,${processedImage}`}
              alt="Processed Preview"
              className="max-w-xs rounded shadow-lg"
            />
          </div>
        )}
        {result && !error && (
          <div className="mt-8 bg-gray-900 text-white p-6 rounded shadow-lg max-w-6xl">
            <h3 className="font-semibold text-center text-lg mb-4">AI Analysis Result:</h3>
            <div className="flex flex-col items-center">
              <span className="text-center">
                <strong>Eye Shape:</strong> {formatDescription(result.eye.shape, result.eye.description)}
              </span>
              <span className="text-center">
                <strong>Eyebrow Shape:</strong> {formatDescription(result.eyebrow.shape, result.eyebrow.description)}
              </span>
              <span className="text-center">
                <strong>Jaw Shape:</strong> {formatDescription(result.jaw.shape, result.jaw.description)}
              </span>
              <span className="text-center">
                <strong>Mouth Shape:</strong> {formatDescription(result.mouth.shape, result.mouth.description)}
              </span>
              <span className="text-center">
                <strong>Nose Shape:</strong> {formatDescription(result.nose.shape, result.nose.description)}
              </span>
            </div>
          </div>
        )}
      </div>
      <div className="mt-8 flex flex-col items-center">
        <h2 className="text-3xl font-bold mb-6 text-center text-white">Upload Your Audio for AI Processing</h2>
        <input
          type="file"
          accept="audio/*"
          onChange={handleAudioChange}
          className="mb-4 p-2 border rounded text-white bg-gray-800"
        />
        {audioPreview && (
          <div className="mt-4">
            <h3 className="font-semibold text-center text-white text-lg">Audio Playback:</h3>
            <audio controls src={audioPreview} className="mt-2"></audio>
          </div>
        )}
        <button
          onClick={handleAudioSubmit}
          className="bg-green-500 text-white py-2 px-6 rounded hover:bg-green-600 transition"
          disabled={audioLoading}
        >
          {audioLoading ? 'Processing Audio...' : 'Process Audio'}
        </button>
        {audioError && (
          <div className="mt-4 bg-red-500 text-white p-4 rounded">
            <p>Error: {audioError}</p>
          </div>
        )}
        {audioResult && !audioError && (
          <div className="mt-8 bg-gray-900 text-white p-6 rounded shadow-lg max-w-6xl">
            <h3 className="font-semibold text-center text-lg mb-4">AI Audio Analysis Result:</h3>
            <ul className="text-center">
              {Object.entries(audioResult).map(([trait, score]) => (
                <li key={trait} className="mb-2">
                  <strong>{trait}:</strong> {score}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
};

export default ImageUploadSection;
