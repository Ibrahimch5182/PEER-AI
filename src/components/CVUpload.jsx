import React, { useState } from "react";

const CVUpload = () => {
  // Mock user state for demo purposes
  const user = { name: "Demo User" };

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
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "application/pdf" || file.type.includes("word"))) {
      setResume(file);
      setResumeName(file.name);
    } else {
      alert("Please upload a valid PDF or DOCX file.");
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf" || file.type.includes("word")) {
        setResume(file);
        setResumeName(file.name);
      } else {
        alert("Please upload a valid PDF or DOCX file.");
      }
    }
  };

  const handleSubmit = async () => {
    if (!user) {
      alert("You must be logged in to use this feature.");
      return;
    }

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

  const formatResumeText = (text) => {
    if (!text) return [];
    
    const cleanText = text.replace(/\[DEBUG\] Full Extracted Resume Text:/, "").trim();
    const lines = cleanText.split("\n").map(line => line.trim()).filter(line => line);
    
    const contactInfo = {
      name: "",
      email: "",
      phone: "",
      linkedin: ""
    };
    
    const emailMatch = cleanText.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/);
    if (emailMatch) contactInfo.email = emailMatch[0];
    
    const phoneMatch = cleanText.match(/(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|(?:\+\d{1,2}[-.\s]?)?\d{10,}/);
    if (phoneMatch) contactInfo.phone = phoneMatch[0];
    
    const linkedinMatch = cleanText.match(/linkedin\.com\/in\/[a-zA-Z0-9-]+/);
    if (linkedinMatch) contactInfo.linkedin = linkedinMatch[0];
    
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

  const personalityTraits = [
    {
      name: "Openness",
      value: openness,
      setter: setOpenness,
      description: "Your curiosity, creativity, and willingness to try new experiences",
      icon: "🎨",
      color: "from-purple-500 to-pink-500"
    },
    {
      name: "Conscientiousness", 
      value: conscientiousness,
      setter: setConscientiousness,
      description: "Your organization, discipline, and attention to detail",
      icon: "📋",
      color: "from-blue-500 to-cyan-500"
    },
    {
      name: "Extraversion",
      value: extraversion,
      setter: setExtraversion,
      description: "Your energy in social situations and preference for interaction",
      icon: "🎭",
      color: "from-orange-500 to-red-500"
    },
    {
      name: "Agreeableness",
      value: agreeableness,
      setter: setAgreeableness,
      description: "Your tendency to be cooperative, trusting, and empathetic",
      icon: "🤝",
      color: "from-green-500 to-teal-500"
    },
    {
      name: "Neuroticism",
      value: neuroticism,
      setter: setNeuroticism,
      description: "Your emotional stability and tendency to experience stress",
      icon: "⚡",
      color: "from-yellow-500 to-orange-500"
    }
  ];

  return (
    <section className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-16" id="cv-upload">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            AI-Powered CV Analysis
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Upload your resume and let our advanced AI analyze your personality traits and career potential
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          {/* File Upload Section */}
          <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 mb-8 border border-white/10">
            <h3 className="text-2xl font-semibold text-white mb-6 flex items-center gap-3">
              <span className="text-3xl">📄</span>
              Upload Your Resume
            </h3>
            
            <div
              className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
                dragActive 
                  ? 'border-blue-400 bg-blue-400/10' 
                  : resume 
                    ? 'border-green-400 bg-green-400/10' 
                    : 'border-gray-400 hover:border-blue-400 hover:bg-blue-400/5'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept=".pdf,.docx"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                onChange={handleFileChange}
              />
              
              <div className="pointer-events-none">
                {resume ? (
                  <div className="text-green-400">
                    <div className="text-6xl mb-4">✅</div>
                    <p className="text-xl font-semibold mb-2">File Selected!</p>
                    <p className="text-gray-300">{resumeName}</p>
                  </div>
                ) : (
                  <div className="text-gray-400">
                    <div className="text-6xl mb-4">📁</div>
                    <p className="text-xl font-semibold mb-2">Drop your resume here</p>
                    <p className="text-gray-300">or click to browse</p>
                    <p className="text-sm text-gray-500 mt-2">Supports PDF and DOCX files</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Personal Information */}
          <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 mb-8 border border-white/10">
            <h3 className="text-2xl font-semibold text-white mb-6 flex items-center gap-3">
              <span className="text-3xl">👤</span>
              Personal Information
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="block text-white font-medium">Gender</label>
                <select 
                  value={gender} 
                  onChange={(e) => setGender(parseInt(e.target.value))}
                  className="w-full p-4 rounded-xl bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-sm"
                >
                  <option value={1} className="bg-gray-800">Male</option>
                  <option value={0} className="bg-gray-800">Female</option>
                </select>
              </div>
              
              <div className="space-y-2">
                <label className="block text-white font-medium">Age</label>
                <input 
                  type="number" 
                  value={age} 
                  onChange={(e) => setAge(parseInt(e.target.value))}
                  className="w-full p-4 rounded-xl bg-white/10 border border-white/20 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent backdrop-blur-sm"
                  min="18"
                  max="100"
                />
              </div>
            </div>
          </div>

          {/* Personality Traits */}
          <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 mb-8 border border-white/10">
            <h3 className="text-2xl font-semibold text-white mb-6 flex items-center gap-3">
              <span className="text-3xl">🧠</span>
              Personality Assessment
            </h3>
            <p className="text-gray-300 mb-8">Rate yourself on each trait from 1-10 to help our AI provide more accurate analysis</p>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {personalityTraits.map((trait, index) => (
                <div key={trait.name} className="bg-white/5 rounded-2xl p-6 border border-white/10">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-2xl">{trait.icon}</span>
                    <div>
                      <h4 className="text-lg font-semibold text-white">{trait.name}</h4>
                      <p className="text-sm text-gray-400">{trait.description}</p>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Score:</span>
                      <span className="text-2xl font-bold text-white bg-gradient-to-r {trait.color} bg-clip-text text-transparent">
                        {trait.value}/10
                      </span>
                    </div>
                    
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={trait.value}
                      onChange={(e) => trait.setter(parseInt(e.target.value))}
                      className="w-full h-3 bg-white/10 rounded-lg appearance-none cursor-pointer slider"
                      style={{
                        background: `linear-gradient(to right, 
                          rgba(59, 130, 246, 0.8) 0%, 
                          rgba(59, 130, 246, 0.8) ${(trait.value - 1) * 11.11}%, 
                          rgba(255, 255, 255, 0.1) ${(trait.value - 1) * 11.11}%, 
                          rgba(255, 255, 255, 0.1) 100%)`
                      }}
                    />
                    
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Low</span>
                      <span>High</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <div className="text-center mb-8">
            <button 
              onClick={handleSubmit} 
              disabled={loading || !resume}
              className={`relative px-12 py-4 rounded-2xl font-semibold text-lg transition-all duration-300 ${
                loading || !resume
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 transform hover:scale-105 shadow-2xl hover:shadow-blue-500/25'
              }`}
            >
              {loading ? (
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  Processing Your CV...
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <span className="text-xl">🚀</span>
                  Analyze My CV
                </div>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-6 mb-8 backdrop-blur-sm">
              <div className="flex items-center gap-3">
                <span className="text-2xl">❌</span>
                <div>
                  <h4 className="text-red-400 font-semibold">Analysis Failed</h4>
                  <p className="text-red-300">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results Display */}
          {result && result.status === "success" && (
            <div className="bg-white/5 backdrop-blur-lg rounded-3xl p-8 border border-white/10">
              <h3 className="text-3xl font-bold text-center text-white mb-8 flex items-center justify-center gap-3">
                <span className="text-4xl">🎯</span>
                Your AI Analysis Results
              </h3>
              
              {/* Personality Traits Prediction */}
              <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl p-6 mb-6 border border-blue-500/20">
                <h4 className="text-xl font-semibold text-blue-400 mb-4 flex items-center gap-2">
                  <span className="text-2xl">🔍</span>
                  Predicted Personality Traits
                </h4>
                <p className="text-gray-300 leading-relaxed">{result.llm_analysis}</p>
              </div>

              {/* Personality Type */}
              <div className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 rounded-2xl p-6 mb-6 border border-yellow-500/20">
                <h4 className="text-xl font-semibold text-yellow-400 mb-4 flex items-center gap-2">
                  <span className="text-2xl">🧠</span>
                  Your Personality Type
                </h4>
                <p className="text-3xl font-bold text-yellow-300 text-center py-4 bg-yellow-500/10 rounded-xl">
                  {result.predicted_personality}
                </p>
              </div>

              {/* Resume Summary */}
              <div className="bg-gradient-to-r from-green-500/10 to-teal-500/10 rounded-2xl p-6 border border-green-500/20">
                <h4 className="text-xl font-semibold text-green-400 mb-4 flex items-center gap-2">
                  <span className="text-2xl">📄</span>
                  Resume Analysis
                </h4>

                {result.resume_data.text ? (
                  (() => {
                    const formattedResume = formatResumeText(result.resume_data.text);
                    
                    return (
                      <div className="space-y-6">
                        {/* Contact Information */}
                        <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                          <h5 className="text-green-400 font-semibold mb-3 flex items-center gap-2">
                            <span>👤</span>
                            Personal Information
                          </h5>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-gray-300">
                            {formattedResume.contactInfo.name && (
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-white">Name:</span>
                                {formattedResume.contactInfo.name}
                              </div>
                            )}
                            
                            {formattedResume.contactInfo.phone && (
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-white">Phone:</span>
                                {formattedResume.contactInfo.phone}
                              </div>
                            )}
                            
                            {formattedResume.contactInfo.email && (
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-white">Email:</span>
                                {formattedResume.contactInfo.email}
                              </div>
                            )}
                            
                            {formattedResume.contactInfo.linkedin && (
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-white">LinkedIn:</span>
                                {formattedResume.contactInfo.linkedin}
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {/* Resume Content */}
                        <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                          <h5 className="text-green-400 font-semibold mb-3 flex items-center gap-2">
                            <span>📝</span>
                            Resume Details
                          </h5>
                          <div className="bg-black/20 rounded-lg p-4 max-h-96 overflow-auto">
                            {formattedResume.lines.map((line, index) => {
                              const isBold = line.length < 30 && 
                                (line.toUpperCase() === line || 
                                 /^(EDUCATION|EXPERIENCE|SKILLS|PROJECTS|WORK|CERTIFICATION)/i.test(line));
                              
                              const needsSpacing = isBold && index > 0;
                              
                              return (
                                <React.Fragment key={index}>
                                  {needsSpacing && <div className="h-4"></div>}
                                  <p className={`mb-1 ${isBold ? 'font-bold text-green-300 text-lg' : 'text-gray-300'}`}>
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
                  <p className="text-gray-300">No resume data available.</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #3b82f6, #8b5cf6);
          cursor: pointer;
          box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: linear-gradient(45deg, #3b82f6, #8b5cf6);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }
      `}</style>
    </section>
  );
};

export default CVUpload;