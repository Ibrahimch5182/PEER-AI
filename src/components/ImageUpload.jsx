import React, { useState } from 'react';

const ImageUploadSection = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);  // State to store the result
  const [processedImage, setProcessedImage] = useState(null);  // State to store the processed image
  const [loading, setLoading] = useState(false);  // State for loading indicator
  const [error, setError] = useState(null);  // State to handle errors

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
    setProcessedImage(null);  // Reset processed image when a new image is selected
    setResult(null);  // Reset results when a new image is selected
    setError(null);  // Reset any previous errors
  };

  const handleSubmit = async () => {
    if (selectedImage) {
      alert(`Image ${selectedImage.name} will be processed!`);
      setLoading(true);
      setError(null);  // Reset any errors before submitting

      // Convert the image to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64String = reader.result.split(',')[1]; // Get the base64 string without the metadata

        // Send the base64 string to the Flask backend
        try {
          const response = await fetch('http://127.0.0.1:5000/api/process-image', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image: base64String, // Send the base64 string
            }),
          });

          const result = await response.json();
          if (result.status === 'success') {
            setResult(result.results);  // Set the result in state
            setProcessedImage(result.image);  // Set the processed image
          } else {
            setError(result.message);  // Set error message if processing fails
          }

        } catch (error) {
          console.error('Error processing image:', error);
          setError('There was an error processing the image.');
        } finally {
          setLoading(false);
        }
      };

      // Read the file as a Data URL (base64)
      reader.readAsDataURL(selectedImage);
    } else {
      alert("Please select an image first.");
    }
  };

  const formatDescription = (shape, description) => {
    // Replace newlines with spaces and trim leading/trailing spaces
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
          disabled={loading}  // Disable button while processing
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
            <div className="flex flex-col items-center"> {/* Center align results */}
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
    </section>
  );
};

export default ImageUploadSection;
