import React, { useState } from 'react';

const Individual = () => {
  const [userData, setUserData] = useState({
    name: "",
    dob: "",
    email: "",
    phone: "",
  });
  const [signResults, setSignResults] = useState(null);
  const [signError, setSignError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [gender, setGender] = useState('Male');
  
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setUserData({ ...userData, [name]: value });
  };

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
    setProcessedImage(null);
    setResult(null);
    setImageError(null);
  };

  const handleGenderChange = (event) => {
    setGender(event.target.value);
  };

  const handleSignSubmit = async () => {
    const dob = new Date(userData.dob);
    const formattedDob = `${dob.getMonth() + 1}/${dob.getDate()}/${dob.getFullYear()}`;
    const { name, email, phone } = userData;

    if (!name || !formattedDob || !email || !phone) {
      alert("Please fill all the fields.");
      return;
    }

    setLoading(true);
    setSignError(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/calculate-signs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dob: formattedDob }),
      });

      const result = await response.json();
      if (result.status === "success") {
        setSignResults(result);
      } else {
        setSignError(result.message);
      }
    } catch (error) {
      console.error("Error fetching Zodiac signs:", error);
      setSignError("There was an error processing your request.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (selectedImage) {
      alert(`Image ${selectedImage.name} will be processed!`);
      setLoading(true);
      setImageError(null);

      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64String = reader.result.split(',')[1];

        try {
          const response = await fetch('http://127.0.0.1:5000/api/find-lookalike', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image: base64String,
              gender: gender,
            }),
          });

          const result = await response.json();
          if (result.status === 'success') {
            setResult(result.matches);
            setProcessedImage(result.image);
          } else {
            setImageError(result.message);
          }
        } catch (error) {
          console.error('Error finding lookalike:', error);
          setImageError('There was an error processing the image.');
        } finally {
          setLoading(false);
        }
      };

      reader.readAsDataURL(selectedImage);
    } else {
      alert('Please select an image first.');
    }
  };

  return (
    <section className="container mx-auto py-16">
      <h2 className="text-3xl font-bold mb-6 text-center text-white">
        User Registration & Zodiac Sign Analysis
      </h2>

      <div className="flex flex-col items-center">
        {/* User Data Inputs */}
        <input
          type="text"
          name="name"
          placeholder="Name"
          value={userData.name}
          onChange={handleInputChange}
          className="mb-4 p-2 border rounded text-white bg-gray-800"
        />
        <input
          type="date"
          name="dob"
          placeholder="Date of Birth"
          value={userData.dob}
          onChange={handleInputChange}
          className="mb-4 p-2 border rounded text-white bg-gray-800"
        />
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={userData.email}
          onChange={handleInputChange}
          className="mb-4 p-2 border rounded text-white bg-gray-800"
        />
        <input
          type="tel"
          name="phone"
          placeholder="Phone"
          value={userData.phone}
          onChange={handleInputChange}
          className="mb-4 p-2 border rounded text-white bg-gray-800"
        />
        <button
          onClick={handleSignSubmit}
          className="bg-purple-500 text-white py-2 px-6 rounded hover:bg-purple-600 transition"
          disabled={loading}
        >
          {loading ? "Processing..." : "Get Zodiac Signs"}
        </button>

        {signError && (
          <div className="mt-4 bg-red-500 text-white p-4 rounded">
            <p>Error: {signError}</p>
          </div>
        )}

        {signResults && !signError && (
          <div className="mt-8 bg-gray-900 text-white p-6 rounded shadow-lg max-w-6xl">
            <h3 className="font-semibold text-center text-lg mb-4">
              Your Zodiac Analysis:
            </h3>
            <p>
              <strong>Zodiac Sign:</strong> {signResults.zodiac_sign.name} -{" "}
              {signResults.zodiac_sign.description}
            </p>
            <p>
              <strong>Chinese Zodiac:</strong> {signResults.chinese_zodiac.name}{" "}
              - {signResults.chinese_zodiac.description}
            </p>
          </div>
        )}

        {/* Image Upload Section */}
        <h2 className="text-3xl font-bold mb-6 text-center text-white">Upload Your Image for AI Processing</h2>

        <div className="flex flex-col items-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="mb-4 p-2 border rounded text-white bg-gray-800"
          />
          <select
            value={gender}
            onChange={handleGenderChange}
            className="mb-4 p-2 border rounded text-white bg-gray-800"
          >
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
          <button
            onClick={handleSubmit}
            className="bg-blue-500 text-white py-2 px-6 rounded hover:bg-blue-600 transition"
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Find Look-Alike'}
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

          {imageError && (
            <div className="mt-4 bg-red-500 text-white p-4 rounded">
              <p>Error: {imageError}</p>
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

          {result && !imageError && (
            <div className="mt-8 bg-gray-900 text-white p-6 rounded shadow-lg max-w-6xl">
              <h3 className="font-semibold text-center text-lg mb-4">Celebrity Look-Alike Matches:</h3>
              {result.length === 0 ? (
                <p>No celebrity matches found.</p>
              ) : (
                result.map((match, index) => (
                  <div key={index} className="mt-4">
                    <h4 className="text-xl font-semibold">{match.name}</h4>
                    <p className="text-sm">Gender: {match.gender}</p>
                    <p>Match Percentage: {match.match_percentage}%</p>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Individual;
