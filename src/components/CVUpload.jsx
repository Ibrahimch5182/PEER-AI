import React, { useState } from "react";

const CVUpload = () => {
  const [resume, setResume] = useState(null);
  const [resumeName, setResumeName] = useState("");
  const [gender, setGender] = useState(1);
  const [age, setAge] = useState(25);
  const [openness, setOpenness] = useState(8);
  const [neuroticism, setNeuroticism] = useState(5);
  const [conscientiousness, setConscientiousness] = useState(7);
  const [agreeableness, setAgreeableness] = useState(6);
  const [extraversion, setExtraversion] = useState(9);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle File Selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "application/pdf" || file.type.includes("word"))) {
      setResume(file);
      setResumeName(file.name);
    } else {
      alert("Please upload a valid PDF or DOCX file.");
    }
  };

  // Handle Submission
  const handleSubmit = async () => {
    if (!resume) {
      alert("Please upload a resume.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("resume", resume);
    formData.append("gender", gender);
    formData.append("age", age);
    formData.append("openness", openness);
    formData.append("neuroticism", neuroticism);
    formData.append("conscientiousness", conscientiousness);
    formData.append("agreeableness", agreeableness);
    formData.append("extraversion", extraversion);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/analyze-cv", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();
      setResult(result);
    } catch (error) {
      console.error("Error analyzing CV:", error);
      setError("There was an error processing the CV.");
    } finally {
      setLoading(false);
    }
  };

  // Format the resume text to look better without complex parsing
  const formatResumeText = (text) => {
    if (!text) return [];
    
    // Clean the text
    const cleanText = text.replace(/\[DEBUG\] Full Extracted Resume Text:/, "").trim();
    
    // Split into lines and filter out empty lines
    const lines = cleanText.split("\n").map(line => line.trim()).filter(line => line);
    
    // Extract basic contact information
    const contactInfo = {
      name: "",
      email: "",
      phone: "",
      linkedin: ""
    };
    
    // Try to extract email
    const emailMatch = cleanText.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/);
    if (emailMatch) contactInfo.email = emailMatch[0];
    
    // Try to extract phone
    const phoneMatch = cleanText.match(/(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|(?:\+\d{1,2}[-.\s]?)?\d{10,}/);
    if (phoneMatch) contactInfo.phone = phoneMatch[0];
    
    // Try to extract LinkedIn
    const linkedinMatch = cleanText.match(/linkedin\.com\/in\/[a-zA-Z0-9-]+/);
    if (linkedinMatch) contactInfo.linkedin = linkedinMatch[0];
    
    // Assume first line might be the name if it's not too long and doesn't contain typical section keywords
    if (lines.length > 0) {
      const firstLine = lines[0];
      if (firstLine.length < 50 && 
          !firstLine.toLowerCase().includes("education") && 
          !firstLine.toLowerCase().includes("experience") && 
          !firstLine.toLowerCase().includes("skills")) {
        contactInfo.name = firstLine.split('|')[0]?.trim() || firstLine;
      }
    }
    
    return {
      contactInfo,
      lines
    };
  };

  return (
    <section className="container mx-auto py-16" id="cv-upload">
      <h2 className="text-3xl font-bold mb-6 text-center text-white">Upload Your CV for AI Analysis</h2>
      
      <div className="flex flex-col items-center">
        {/* File Upload Input */}
        <label className="w-64 flex flex-col items-center px-4 py-6 bg-gray-800 text-white rounded-lg shadow-lg tracking-wide border border-blue-500 cursor-pointer hover:bg-blue-600">
          <span className="text-lg">📄 Choose a Resume</span>
          <input type="file" accept=".pdf,.docx" className="hidden" onChange={handleFileChange} />
        </label>

        {resumeName && <p className="mt-2 text-white text-sm">Selected: {resumeName}</p>}

        {/* Dropdowns and Inputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6 text-white">
          <div>
            <label className="block">Gender:</label>
            <select value={gender} onChange={(e) => setGender(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500 focus:outline-none focus:ring focus:ring-blue-500">
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>
          <div>
            <label className="block">Age:</label>
            <input type="number" value={age} onChange={(e) => setAge(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500 focus:outline-none focus:ring focus:ring-blue-500" />
          </div>
          <div>
            <label className="block">Openness (1-10):</label>
            <input type="number" value={openness} onChange={(e) => setOpenness(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500" />
          </div>
          <div>
            <label className="block">Neuroticism (1-10):</label>
            <input type="number" value={neuroticism} onChange={(e) => setNeuroticism(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500" />
          </div>
          <div>
            <label className="block">Conscientiousness (1-10):</label>
            <input type="number" value={conscientiousness} onChange={(e) => setConscientiousness(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500" />
          </div>
          <div>
            <label className="block">Agreeableness (1-10):</label>
            <input type="number" value={agreeableness} onChange={(e) => setAgreeableness(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500" />
          </div>
          <div>
            <label className="block">Extraversion (1-10):</label>
            <input type="number" value={extraversion} onChange={(e) => setExtraversion(parseInt(e.target.value))}
              className="w-full p-2 rounded bg-gray-700 border border-gray-500" />
          </div>
        </div>

        {/* Submit Button */}
        <button onClick={handleSubmit} disabled={loading}
          className="mt-6 bg-blue-500 text-white py-2 px-6 rounded hover:bg-blue-600 transition">
          {loading ? "Processing..." : "Analyze CV"}
        </button>

        {/* Error Handling */}
        {error && (
          <div className="mt-4 bg-red-500 text-white p-4 rounded">
            <p>Error: {error}</p>
          </div>
        )}

        {/* Display Formatted Result */}
        {result && result.status === "success" && (
          <div className="mt-8 bg-gray-900 text-white p-6 rounded shadow-lg max-w-6xl w-full">
            <h3 className="font-semibold text-center text-lg mb-4">📊 AI Analysis Result</h3>
            
            {/* Personality Traits */}
            <div className="bg-gray-800 p-4 rounded mb-4">
              <h4 className="text-blue-400 text-lg font-semibold">🔍 Predicted Personality Traits:</h4>
              <p className="text-sm text-gray-300">{result.llm_analysis}</p>
            </div>

            {/* Personality Type */}
            <div className="bg-gray-800 p-4 rounded mb-4">
              <h4 className="text-blue-400 text-lg font-semibold">🧠 Predicted Personality Type:</h4>
              <p className="text-lg font-bold text-yellow-300">{result.predicted_personality}</p>
            </div>

            {/* Resume Summary */}
            <div className="bg-gray-800 p-4 rounded">
              <h4 className="text-blue-400 text-lg font-semibold mb-3">📄 Resume Summary:</h4>

              {result.resume_data.text ? (
                (() => {
                  const formattedResume = formatResumeText(result.resume_data.text);
                  
                  return (
                    <div className="space-y-4">
                      {/* Contact Information */}
                      <div className="border-b border-gray-700 pb-3">
                        <h5 className="text-blue-400 font-semibold mb-2">👤 Personal Information</h5>
                        <div className="text-sm text-gray-300 grid grid-cols-1 md:grid-cols-2 gap-2">
                          {formattedResume.contactInfo.name && (
                            <div>
                              <span className="font-medium">Name:</span>{" "}
                              {formattedResume.contactInfo.name}
                            </div>
                          )}
                          
                          {formattedResume.contactInfo.phone && (
                            <div>
                              <span className="font-medium">Contact:</span>{" "}
                              {formattedResume.contactInfo.phone}
                            </div>
                          )}
                          
                          {formattedResume.contactInfo.email && (
                            <div>
                              <span className="font-medium">Email:</span>{" "}
                              {formattedResume.contactInfo.email}
                            </div>
                          )}
                          
                          {formattedResume.contactInfo.linkedin && (
                            <div>
                              <span className="font-medium">LinkedIn:</span>{" "}
                              {formattedResume.contactInfo.linkedin}
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* Resume Content - Enhanced with styling but without complex parsing */}
                      <div className="text-sm text-gray-300">
                        <h5 className="text-blue-400 font-semibold mb-2">📝 Resume Details</h5>
                        <div className="bg-gray-700 p-4 rounded overflow-auto max-h-96">
                          {formattedResume.lines.map((line, index) => {
                            // Check if this line might be a section header
                            const isBold = line.length < 30 && 
                              (line.toUpperCase() === line || 
                               /^(EDUCATION|EXPERIENCE|SKILLS|PROJECTS|WORK|CERTIFICATION)/i.test(line));
                            
                            // Apply spacing before headers for better visual separation
                            const needsSpacing = isBold && index > 0;
                            
                            return (
                              <React.Fragment key={index}>
                                {needsSpacing && <div className="h-4"></div>}
                                <p className={`mb-1 ${isBold ? 'font-bold text-blue-300' : ''}`}>
                                  {line}
                                </p>
                              </React.Fragment>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  );
                })()
              ) : (
                <p className="text-sm text-gray-300">No resume data available.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default CVUpload;