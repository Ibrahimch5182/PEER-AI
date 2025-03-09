import React, { useState } from 'react';

const LinkedInAnalysis = () => {
  const [profileUrl, setProfileUrl] = useState('');
  const [jobPreferences, setJobPreferences] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [profileImage, setProfileImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [followUpInput, setFollowUpInput] = useState('');

  const handleUrlChange = (event) => {
    setProfileUrl(event.target.value);
    setAnalysis(null);
    setProfileImage(null);
    setError(null);
    setMessages([]);
  };

  const handleJobPreferencesChange = (event) => {
    setJobPreferences(event.target.value);
  };

  const handleSubmit = async () => {
    if (!profileUrl) {
      alert('Please enter a LinkedIn profile URL.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/analyze-linkedin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_url: profileUrl, job_preferences: jobPreferences }),
      });

      const result = await response.json();

      if (result.status === 'success') {
        setAnalysis(result.analysis);
        setProfileImage(result.profile_image_url);
        setMessages([{ role: 'assistant', content: result.analysis }]);
      } else {
        setError(result.message || 'Failed to analyze the profile.');
      }
    } catch (error) {
      console.error('Error analyzing LinkedIn profile:', error);
      setError('There was an error processing the LinkedIn profile.');
    } finally {
      setLoading(false);
    }
  };

  const handleFollowUpSubmit = async () => {
    if (!followUpInput) {
      alert('Please enter a follow-up question.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/analyze-linkedin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: followUpInput }),
      });

      const result = await response.json();

      if (result.status === 'success') {
        setMessages((prevMessages) => [
          ...prevMessages,
          { role: 'user', content: followUpInput },
          { role: 'assistant', content: result.analysis },
        ]);
        setFollowUpInput('');
      } else {
        setError(result.message || 'Failed to process follow-up question.');
      }
    } catch (error) {
      console.error('Error processing follow-up question:', error);
      setError('There was an error processing your question.');
    } finally {
      setLoading(false);
    }
  };

  // Function to extract rating from analysis text
  const extractRating = (text) => {
    const ratingRegex = /Rating:\s*(\d+(?:\/|\.)\d+|\d+)/i;
    const match = text.match(ratingRegex);
    return match ? match[1] : null;
  };

  // Function to parse analysis into key sections
  const parseAnalysis = (text) => {
    if (!text) return null;
    
    // Extract rating
    const rating = extractRating(text);
    
    // Split into sections based on numbered points
    const sections = [];
    const mainParts = text.split(/\d+\.\s+/);
    
    // The first part is usually the introduction with rating
    const intro = mainParts[0].replace(/Rating:.*$/m, '').trim();
    
    // Extract section titles and content
    for (let i = 1; i < mainParts.length; i++) {
      const part = mainParts[i];
      const titleMatch = part.match(/^([^:]+):/);
      
      if (titleMatch) {
        const title = titleMatch[1].trim();
        const content = part.replace(/^[^:]+:/, '').trim();
        sections.push({ title, content });
      } else {
        sections.push({ title: `Section ${i}`, content: part.trim() });
      }
    }
    
    return {
      rating,
      intro,
      sections
    };
  };
  
  // Function to determine rating color
  const getRatingColor = (rating) => {
    if (!rating) return 'text-gray-400';
    
    const numericRating = parseFloat(rating.replace('/', '.'));
    if (numericRating >= 8) return 'text-green-400';
    if (numericRating >= 6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-gray-900 to-blue-900 text-white">
      {/* Sidebar */}
      <div className="w-1/4 p-6 border-r border-gray-700 bg-gray-800 bg-opacity-80 backdrop-blur-sm">
        <div className="sticky top-0">
          <h1 className="flex items-center text-2xl font-bold mb-8">
            <span className="bg-purple-600 p-2 rounded mr-2">Peer-In</span>
            <img
              src="https://www.pagetraffic.com/blog/wp-content/uploads/2022/09/linkedin-blue-logo-icon.png"
              alt="LinkedIn Logo"
              className="w-6 h-6 ml-1"
            />
          </h1>

          {/* Features and Tips */}
          <div className="mb-6">
            <details className="p-3 bg-gray-700 bg-opacity-50 rounded-lg mb-4">
              <summary className="cursor-pointer font-semibold flex items-center">
                <span className="text-purple-400 mr-2">🚀</span> Features and Tips
              </summary>
              <ul className="mt-3 pl-4 space-y-2 text-sm">
                <li className="flex items-start">
                  <span className="text-purple-400 mr-2">🧐</span> 
                  <span><strong>Profile Review:</strong> Personalized, actionable advice for enhancing your profile.</span>
                </li>
                <li className="flex items-start">
                  <span className="text-purple-400 mr-2">📚</span> 
                  <span><strong>Specialized Knowledge:</strong> Custom knowledge base for LinkedIn improvements.</span>
                </li>
                <li className="flex items-start">
                  <span className="text-purple-400 mr-2">💡</span> 
                  <span><strong>Local LLM Power:</strong> Local LLaMA model for analysis.</span>
                </li>
                <li className="flex items-start">
                  <span className="text-purple-400 mr-2">🔥</span> 
                  <span><strong>Tip:</strong> Set your profile to public for better analysis.</span>
                </li>
              </ul>
            </details>
          </div>

          {/* Job Preferences */}
          <div className="mb-6">
            <details className="p-3 bg-gray-700 bg-opacity-50 rounded-lg mb-4">
              <summary className="cursor-pointer font-semibold flex items-center">
                <span className="text-purple-400 mr-2">🎯</span> Job Preferences & Context
              </summary>
              <textarea
                value={jobPreferences}
                onChange={handleJobPreferencesChange}
                placeholder="e.g., 'Software Engineer, entry-level, Tech industry and interested in AI.'"
                className="w-full p-3 mt-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none"
                rows={4}
              />
            </details>
          </div>

          {/* LinkedIn Profile URL Input */}
          <div className="mb-6">
            <div className="relative">
              <input
                type="text"
                placeholder="LinkedIn Profile URL"
                value={profileUrl}
                onChange={handleUrlChange}
                className="w-full p-3 pl-10 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none"
              />
              <span className="absolute left-3 top-3 text-purple-400">🌐</span>
            </div>
          </div>

          {/* Analyze Button */}
          <button
            onClick={handleSubmit}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-500 text-white py-3 px-6 rounded-lg hover:from-purple-700 hover:to-blue-600 transition duration-300 font-semibold flex justify-center items-center"
            disabled={loading}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </>
            ) : (
              'Analyze Profile'
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="mt-4 bg-red-500 bg-opacity-80 text-white p-4 rounded-lg">
              <p>Error: {error}</p>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="w-3/4 p-6">
        {!analysis && !loading && (
          <div className="text-center flex flex-col items-center justify-center h-full">
            <h2 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">
              Ready to enhance your LinkedIn profile?
            </h2>
            <p className="text-xl mb-8 max-w-2xl">
              Drop your 
              <img
                src="https://www.pagetraffic.com/blog/wp-content/uploads/2022/09/linkedin-blue-logo-icon.png"
                alt="LinkedIn Logo"
                className="inline-block w-6 h-6 mx-2"
              /> 
              profile URL in the sidebar and let's dive in!
            </p>
            
            <div className="relative h-64 w-64 mt-8">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-blue-400 rounded-full opacity-30 animate-pulse"></div>
              <div className="absolute inset-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                  <rect x="2" y="9" width="4" height="12"></rect>
                  <circle cx="4" cy="4" r="2"></circle>
                </svg>
              </div>
            </div>
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="w-16 h-16 border-t-4 border-purple-500 border-solid rounded-full animate-spin"></div>
            <p className="mt-6 text-xl">Analyzing your profile...</p>
            <p className="text-gray-400">This may take a moment</p>
          </div>
        )}

        {analysis && (
          <div className="mt-4">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Profile Image Section */}
              {profileImage && (
                <div className="lg:col-span-3">
                  <div className="bg-gray-800 bg-opacity-70 rounded-lg p-4 backdrop-blur-sm">
                    <div className="flex flex-col items-center">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full opacity-70 animate-pulse"></div>
                        <img
                          src={profileImage}
                          alt="LinkedIn Profile"
                          className="relative rounded-full w-32 h-32 object-cover border-4 border-gray-700"
                        />
                      </div>
                      
                      {messages[0]?.content && (
                        <div className="mt-4 w-full">
                          <div className="flex justify-center items-center">
                            <h3 className="text-lg font-semibold mr-2">Rating:</h3>
                            <span className={`text-xl font-bold ${getRatingColor(extractRating(messages[0].content))}`}>
                              {extractRating(messages[0].content) || "N/A"}
                            </span>
                          </div>
                          
                          {/* Rating visualization */}
                          <div className="mt-2 w-full bg-gray-700 rounded-full h-2.5">
                            <div 
                              className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-2.5 rounded-full" 
                              style={{ 
                                width: `${extractRating(messages[0].content)?.replace('/10', '') * 10}%` 
                              }}
                            ></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              {/* Analysis Content */}
              <div className={`${profileImage ? 'lg:col-span-9' : 'lg:col-span-12'}`}>
                {messages[0]?.content && (
                  <div className="bg-gray-800 bg-opacity-70 rounded-lg p-6 backdrop-blur-sm">
                    {(() => {
                      const parsedAnalysis = parseAnalysis(messages[0].content);
                      if (!parsedAnalysis) return <p>{messages[0].content}</p>;
                      
                      return (
                        <div>
                          <h2 className="text-2xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">
                            LinkedIn Profile Analysis
                          </h2>
                          
                          {/* Introduction */}
                          <div className="mb-6">
                            <p className="text-gray-300">{parsedAnalysis.intro}</p>
                          </div>
                          
                          {/* Sections */}
                          <div className="space-y-6">
                            {parsedAnalysis.sections.map((section, index) => (
                              <div key={index} className="bg-gray-700 bg-opacity-50 rounded-lg p-4">
                                <h3 className="text-lg font-semibold mb-2 flex items-center">
                                  <span className="text-purple-400 mr-2">{index + 1}.</span>
                                  {section.title}
                                </h3>
                                <p className="text-gray-300">{section.content}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
                
                {/* Additional messages */}
                {messages.slice(1).map((message, index) => (
                  <div
                    key={index + 1}  // +1 because we've already rendered messages[0]
                    className={`mt-4 p-4 rounded-lg ${
                      message.role === "user" 
                        ? "bg-purple-900 bg-opacity-50 ml-12" 
                        : "bg-gray-800 bg-opacity-70 mr-12"
                    }`}
                  >
                    <p>{message.content}</p>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Follow-up Questions */}
            {analysis && (
              <div className="mt-8">
                <div className="bg-gray-800 bg-opacity-70 rounded-lg p-4 backdrop-blur-sm">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Ask me anything about improving your LinkedIn profile!"
                      value={followUpInput}
                      onChange={(e) => setFollowUpInput(e.target.value)}
                      className="w-full p-4 pl-12 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none"
                      onKeyPress={(e) => e.key === 'Enter' && handleFollowUpSubmit()}
                    />
                    <span className="absolute left-4 top-4 text-purple-400">💬</span>
                    <button
                      onClick={handleFollowUpSubmit}
                      className="absolute right-3 top-3 bg-gradient-to-r from-purple-500 to-blue-500 text-white py-1 px-4 rounded-lg hover:from-purple-600 hover:to-blue-600 transition duration-300"
                      disabled={loading}
                    >
                      {loading ? 'Sending...' : 'Send'}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default LinkedInAnalysis;