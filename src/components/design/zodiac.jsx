import React, { useState } from "react";

const Zodiac = () => {
  const [userData, setUserData] = useState({
    name: "",
    dob: "",
    email: "",
    phone: "",
  });
  const [selectedImage, setSelectedImage] = useState(null);
  const [gender, setGender] = useState("Male");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setUserData({ ...userData, [name]: value });
  };

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
  };

  const handleGenderChange = (event) => {
    setGender(event.target.value);
  };

  const handleSubmit = async () => {
    const { name, dob, email, phone } = userData;

    if (!name || !dob || !email || !phone || !selectedImage || !gender) {
      alert("Please fill all fields and upload an image.");
    
//    if (response.status !== 200) {
//      throw new Error("Failed to process the request.");
//      }
      
      return;
    }

    setLoading(true);
    setError(null);

    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Image = reader.result.split(",")[1]; // Get base64 string of image

      try {
        const response = await fetch("http://127.0.0.1:5000/api/process-all", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            email,
            dob,
            number: phone,
            image: base64Image,
            gender,
          }),
        });

        const result = await response.json();
        if (result.status === "success") {
          setResults(result);
        } else {
          setError(result.message);
        }
      } catch (error) {
        console.error("Error during API call:", error);
        setError("An unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    };

    reader.readAsDataURL(selectedImage); // Read image as base64
  };

  const handleDownloadPDF = () => {
    if (results?.pdf_url) {
      const link = document.createElement("a");
      link.href = `http://127.0.0.1:5000${results.pdf_url}`; // Use the server-hosted URL
      link.download = "zodiac_report.pdf";
      link.click();
    }
  };
  
  

  return (
    <section className="container mx-auto py-16">
      <h2 className="text-3xl font-bold mb-6 text-center text-white">
        User Registration & Full Analysis
      </h2>

      <div className="flex flex-col items-center">
        {/* Input Fields */}
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
          className="bg-purple-500 text-white py-2 px-6 rounded hover:bg-purple-600 transition"
          disabled={loading}
        >
          {loading ? "Processing..." : "Submit"}
        </button>

        {/* Results Section */}
        {error && (
          <div className="mt-4 bg-red-500 text-white p-4 rounded">
            <p>Error: {error}</p>
          </div>
        )}
        {results && !error && (
          <div className="mt-8 bg-gray-900 text-white p-6 rounded shadow-lg max-w-6xl">
            <h3 className="font-semibold text-center text-lg mb-4">Analysis Results:</h3>
            <p><strong>Zodiac Sign:</strong> {results.zodiac_results.zodiac_sign.name}</p>
            <p><strong>Chinese Zodiac:</strong> {results.zodiac_results.chinese_zodiac}</p>
            <p><strong>Facial Features:</strong></p>
            <ul>
              {Object.entries(results.facial_features).map(([key, value]) => (
                <li key={key}>{key}: {value}</li>
              ))}
            </ul>
            <p><strong>Celebrity Matches:</strong></p>
            <ul>
              {results.lookalikes.map((match, index) => (
                <li key={index}>{match.name} ({match.match_percentage}%)</li>
              ))}
            </ul>
            <button
              onClick={handleDownloadPDF}
              className="bg-blue-500 text-white py-2 px-6 rounded hover:bg-blue-600 transition mt-4"
            >
              Download PDF Report
            </button>
          </div>
        )}
      </div>
    </section>
  );
};

export default Zodiac;
