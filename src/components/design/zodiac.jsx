import React, { useState } from "react";

const Zodiac = () => {
  const [step, setStep] = useState(1);
  const [userData, setUserData] = useState({
    name: "",
    dob: "",
    email: "",
    phone: "",
  });
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [gender, setGender] = useState("Male");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setUserData({ ...userData, [name]: value });
  };

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenderChange = (event) => {
    setGender(event.target.value);
  };

  const nextStep = () => {
    // Validate current step
    if (step === 1) {
      if (!userData.name || !userData.dob) {
        setError("Please provide your name and date of birth");
        return;
      }
    } else if (step === 2) {
      if (!userData.email || !userData.phone) {
        setError("Please provide your email and phone number");
        return;
      }
    } else if (step === 3) {
      if (!selectedImage) {
        setError("Please upload your photo");
        return;
      }
    }
    
    setError(null);
    setStep(step + 1);
  };

  const prevStep = () => {
    setStep(step - 1);
    setError(null);
  };

  const validateEmail = (email) => {
    const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
  };

  const validatePhone = (phone) => {
    // Simple validation - can be replaced with more specific requirements
    return phone.length >= 10;
  };

  const handleSubmit = async () => {
    // Final validation
    if (!userData.name || !userData.dob || !userData.email || !userData.phone || !selectedImage) {
      setError("Please complete all fields before submitting");
      return;
    }

    if (!validateEmail(userData.email)) {
      setError("Please enter a valid email address");
      return;
    }

    if (!validatePhone(userData.phone)) {
      setError("Please enter a valid phone number");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const base64Image = await toBase64(selectedImage);

      const response = await fetch("http://127.0.0.1:5000/api/process-all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: userData.name,
          email: userData.email,
          dob: userData.dob,
          number: userData.phone,
          image: base64Image,
          gender,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to process the request.");
      }

      const result = await response.json();
      if (result.status === "success") {
        setResults(result);
        setStep(5); // Move to results page
      } else {
        setError(result.message || "An error occurred processing your data");
      }
    } catch (error) {
      console.error("Error during API call:", error);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
  try {
    setLoading(true);
    const response = await fetch("http://127.0.0.1:5000/api/process-all", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: userData.name,
        email: userData.email,
        dob: userData.dob,
        number: userData.phone,
        gender: gender,
        image: await toBase64(selectedImage),
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to process request");
    }

    const result = await response.json();
    
    if (result.status === "success") {
      // Create a link and trigger download from the file path
      const downloadLink = `http://127.0.0.1:5000${result.file_path}`;
      const link = document.createElement("a");
      link.href = downloadLink;
      link.download = `${userData.name}_Personality_Report.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Update the results state for display
      setResults(result);
    } else {
      setError(result.message || "An error occurred processing your data");
    }
  } catch (error) {
    console.error("Error downloading PDF:", error);
    setError("Could not download your report. Please try again.");
  } finally {
    setLoading(false);
  }
};
  // Helper function to convert an image file to base64
  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  // Calculate birth date information
  const getBirthInfo = () => {
    if (!userData.dob) return null;
    
    const birthDate = new Date(userData.dob);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    
    const days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const months = ["January", "February", "March", "April", "May", "June", "July", 
                   "August", "September", "October", "November", "December"];
    
    return {
      age,
      day: days[birthDate.getDay()],
      date: birthDate.getDate(),
      month: months[birthDate.getMonth()],
      year: birthDate.getFullYear()
    };
  };

  const birthInfo = getBirthInfo();

  return (
    <section className="min-h-screen bg-gradient-to-b from-gray-900 to-purple-900 text-white py-12 px-4">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-3 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">
            Cosmic Personality Analysis
          </h1>
          <p className="text-lg text-gray-300">
            Discover your celestial profile and cosmic connections
          </p>
        </div>

        {/* Progress bar */}
        <div className="mb-10">
          <div className="flex justify-between mb-2">
            {["Personal", "Contact", "Photo", "Confirm", "Results"].map((label, index) => (
              <div 
                key={index} 
                className={`text-xs md:text-sm ${step > index ? "text-purple-400" : "text-gray-400"}`}
              >
                {label}
              </div>
            ))}
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5">
            <div 
              className="bg-gradient-to-r from-purple-500 to-pink-500 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${(step / 5) * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 bg-red-900 border border-red-500 text-white p-4 rounded-lg flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p>{error}</p>
          </div>
        )}

        {/* Form container with glass effect */}
        <div className="bg-gray-900 bg-opacity-60 backdrop-filter backdrop-blur-lg p-8 rounded-2xl shadow-xl border border-gray-800">
          {/* Step 1: Personal Info */}
          {step === 1 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold mb-6 text-center">Tell Us About Yourself</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-300">Your Full Name</label>
                  <input
                    type="text"
                    name="name"
                    placeholder="Enter your full name"
                    value={userData.name}
                    onChange={handleInputChange}
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-300">Date of Birth</label>
                  <input
                    type="date"
                    name="dob"
                    value={userData.dob}
                    onChange={handleInputChange}
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-300">Gender</label>
                  <div className="flex space-x-4">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="gender"
                        value="Male"
                        checked={gender === "Male"}
                        onChange={handleGenderChange}
                        className="mr-2 text-purple-500 focus:ring-purple-500"
                      />
                      <span>Male</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="gender"
                        value="Female"
                        checked={gender === "Female"}
                        onChange={handleGenderChange}
                        className="mr-2 text-purple-500 focus:ring-purple-500"
                      />
                      <span>Female</span>
                    </label>
                  </div>
                </div>
              </div>
              
              <div className="flex justify-end mt-8">
                <button
                  onClick={nextStep}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-medium py-2 px-6 rounded-lg transition transform hover:scale-105 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50"
                >
                  Continue
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Contact Info */}
          {step === 2 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold mb-6 text-center">Your Contact Information</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-300">Email Address</label>
                  <input
                    type="email"
                    name="email"
                    placeholder="your.email@example.com"
                    value={userData.email}
                    onChange={handleInputChange}
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-300">Phone Number</label>
                  <input
                    type="tel"
                    name="phone"
                    placeholder="Your phone number"
                    value={userData.phone}
                    onChange={handleInputChange}
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition text-white"
                  />
                </div>
              </div>
              
              <div className="flex justify-between mt-8">
                <button
                  onClick={prevStep}
                  className="bg-gray-700 hover:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition"
                >
                  Back
                </button>
                <button
                  onClick={nextStep}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-medium py-2 px-6 rounded-lg transition transform hover:scale-105 hover:shadow-lg"
                >
                  Continue
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Photo Upload */}
          {step === 3 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold mb-6 text-center">Upload Your Photo</h2>
              
              <div className="space-y-6">
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center">
                  {imagePreview ? (
                    <div className="flex flex-col items-center">
                      <img 
                        src={imagePreview} 
                        alt="Preview" 
                        className="max-h-64 rounded-lg mb-4" 
                      />
                      <button
                        onClick={() => {
                          setSelectedImage(null);
                          setImagePreview(null);
                        }}
                        className="text-red-400 hover:text-red-300 text-sm"
                      >
                        Remove photo
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <p className="text-gray-400 mb-4">Drag and drop your photo or click to browse</p>
                      <label className="bg-purple-600 hover:bg-purple-700 text-white font-medium py-2 px-4 rounded-lg cursor-pointer transition">
                        Select Photo
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleImageChange}
                          className="hidden"
                        />
                      </label>
                    </div>
                  )}
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg text-sm text-gray-300">
                  <p className="font-medium mb-2">Photo guidelines:</p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Use a clear, recent front-facing photo</li>
                    <li>Ensure good lighting with no shadows on your face</li>
                    <li>Remove sunglasses or other items that cover your face</li>
                    <li>Neutral expression works best for accurate analysis</li>
                  </ul>
                </div>
              </div>
              
              <div className="flex justify-between mt-8">
                <button
                  onClick={prevStep}
                  className="bg-gray-700 hover:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition"
                >
                  Back
                </button>
                <button
                  onClick={nextStep}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-medium py-2 px-6 rounded-lg transition transform hover:scale-105 hover:shadow-lg"
                >
                  Continue
                </button>
              </div>
            </div>
          )}

          {/* Step 4: Confirmation */}
          {step === 4 && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold mb-6 text-center">Confirm Your Information</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-800 p-4 rounded-lg">
                  <h3 className="font-medium text-purple-400 mb-2">Personal Details</h3>
                  <p><span className="text-gray-400">Name:</span> {userData.name}</p>
                  {birthInfo && (
                    <>
                      <p><span className="text-gray-400">Birthday:</span> {birthInfo.month} {birthInfo.date}, {birthInfo.year}</p>
                      <p><span className="text-gray-400">Day of birth:</span> {birthInfo.day}</p>
                      <p><span className="text-gray-400">Age:</span> {birthInfo.age} years</p>
                    </>
                  )}
                  <p><span className="text-gray-400">Gender:</span> {gender}</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg">
                  <h3 className="font-medium text-purple-400 mb-2">Contact Information</h3>
                  <p><span className="text-gray-400">Email:</span> {userData.email}</p>
                  <p><span className="text-gray-400">Phone:</span> {userData.phone}</p>
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg md:col-span-2">
                  <h3 className="font-medium text-purple-400 mb-2">Your Photo</h3>
                  <div className="flex justify-center">
                    {imagePreview && (
                      <img src={imagePreview} alt="Your uploaded photo" className="max-h-40 rounded" />
                    )}
                  </div>
                </div>
              </div>
              
              <div className="bg-indigo-900 bg-opacity-50 border border-indigo-800 p-4 rounded-lg text-sm">
                <p>By clicking Submit, you agree to our analysis of your data for the purpose of generating your cosmic profile. Your information will be processed securely and shared according to our privacy policy.</p>
              </div>
              
              <div className="flex justify-between mt-8">
                <button
                  onClick={prevStep}
                  className="bg-gray-700 hover:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition"
                >
                  Back
                </button>
                <button
                  onClick={handleSubmit}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-medium py-2 px-6 rounded-lg transition transform hover:scale-105 hover:shadow-lg flex items-center"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </>
                  ) : (
                    "Submit Analysis"
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Step 5: Results */}
          {step === 5 && results && (
            <div className="space-y-8">
              <div className="text-center">
                <h2 className="text-3xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">
                  Your Cosmic Profile
                </h2>
                <p className="text-gray-300">
                  Based on your birth data and facial features
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Zodiac Sign Section */}
                <div className="bg-gray-800 bg-opacity-70 p-6 rounded-xl shadow-lg border border-gray-700">
                  <h3 className="text-xl font-semibold mb-4 text-center text-purple-400">
                    Zodiac Analysis
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-center">
                      <div className="w-10 h-10 flex-shrink-0 mr-3 bg-purple-800 rounded-full flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm text-gray-400">Western Zodiac</span>
                        <p className="font-medium">{results.zodiac_results.zodiac_sign.name}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center">
                      <div className="w-10 h-10 flex-shrink-0 mr-3 bg-purple-800 rounded-full flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm text-gray-400">Chinese Zodiac</span>
                        <p className="font-medium">{results.zodiac_results.chinese_zodiac}</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Facial Features Section */}
                <div className="bg-gray-800 bg-opacity-70 p-6 rounded-xl shadow-lg border border-gray-700">
                  <h3 className="text-xl font-semibold mb-4 text-center text-purple-400">
                    Facial Analysis
                  </h3>
                  <ul className="space-y-2">
                    {Object.entries(results.facial_features).map(([key, value]) => (
                      <li key={key} className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-purple-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                        <span className="capitalize text-gray-300">{key.replace(/_/g, ' ')}:</span>
                        <span className="ml-1">{value}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              
              {/* Celebrity Lookalikes */}
              <div className="bg-gray-800 bg-opacity-70 p-6 rounded-xl shadow-lg border border-gray-700">
                <h3 className="text-xl font-semibold mb-4 text-center text-purple-400">
                  Celebrity Matches
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {results.lookalikes.map((match, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-3 flex items-center">
                      <div className="w-12 h-12 bg-purple-900 rounded-full flex items-center justify-center text-lg font-bold mr-3">
                        {match.match_percentage}%
                      </div>
                      <div>
                        <p className="font-medium">{match.name}</p>
                        <p className="text-xs text-gray-400">Match Score</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Download Report Button */}
              <div className="flex justify-center mt-8">
                <button
                  onClick={handleDownloadPDF}
                  className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium py-3 px-8 rounded-lg transition transform hover:scale-105 hover:shadow-lg flex items-center"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Generating...
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                      Download Full Report
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Zodiac;