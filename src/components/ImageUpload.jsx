import React, { useState } from 'react';
import { Upload, Image as ImageIcon, Music, Play, Pause, Eye, Users, Zap, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const ImageUploadSection = () => {
  // Mock user for demo - replace with your AuthContext
  const user = { name: 'Demo User' };
  
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);
  const [audioResult, setAudioResult] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dragOver, setDragOver] = useState({ image: false, audio: false });

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setProcessedImage(null);
      setResult(null);
      setError(null);
    }
  };

  const handleAudioChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("audio/")) {
      setSelectedAudio(file);
      setAudioResult(null);
      setAudioError(null);
      setAudioPreview(URL.createObjectURL(file));
    } else {
      alert("Please upload a valid audio file.");
    }
  };

  const handleDragOver = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: true }));
  };

  const handleDragLeave = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: false }));
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: false }));
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (type === 'image' && file.type.startsWith('image/')) {
        setSelectedImage(file);
        setProcessedImage(null);
        setResult(null);
        setError(null);
      } else if (type === 'audio' && file.type.startsWith('audio/')) {
        setSelectedAudio(file);
        setAudioResult(null);
        setAudioError(null);
        setAudioPreview(URL.createObjectURL(file));
      }
    }
  };

  const handleSubmit = async () => {
    if (!user) {
      alert("You must be logged in to use this feature.");
      return;
    }

    if (!selectedImage) {
      alert('Please select an image first.');
      return;
    }

    setLoading(true);
    setError(null);

    // Simulate API call
    setTimeout(() => {
      const mockResult = {
        eye: { shape: "Almond", description: "Well-defined almond-shaped eyes with natural symmetry" },
        eyebrow: { shape: "Arched", description: "Naturally arched eyebrows with good definition" },
        jaw: { shape: "Oval", description: "Balanced oval jawline with soft contours" },
        mouth: { shape: "Full", description: "Well-proportioned lips with natural fullness" },
        nose: { shape: "Straight", description: "Classic straight nose bridge with refined tip" }
      };
      
      setResult(mockResult);
      setProcessedImage("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==");
      setLoading(false);
    }, 3000);
  };

  const handleAudioSubmit = async () => {
    if (!user) {
      alert("You must be logged in to use this feature.");
      return;
    }

    if (!selectedAudio) {
      alert('Please select an audio file first.');
      return;
    }

    setAudioLoading(true);
    setAudioError(null);

    // Simulate API call
    setTimeout(() => {
      const mockResult = {
        "Confidence": "85%",
        "Emotional Tone": "Calm & Professional",
        "Speech Clarity": "Excellent",
        "Vocal Strength": "Strong",
        "Communication Style": "Articulate"
      };
      
      setAudioResult(mockResult);
      setAudioLoading(false);
    }, 2500);
  };

  const formatDescription = (shape, description) => {
    return `${shape} - ${description.replace(/\n/g, ' ').trim()}`;
  };

  const FeatureCard = ({ icon: Icon, title, description, color }) => (
    <div className="group bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 hover:border-gray-600/50 transition-all duration-300 hover:transform hover:scale-105">
      <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
        <Icon className="w-6 h-6 text-white" />
      </div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );

  const AnalysisCard = ({ title, items, icon: Icon, gradient }) => (
    <div className={`bg-gradient-to-br ${gradient} rounded-xl p-6 backdrop-blur-sm border border-white/10 shadow-2xl`}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-lg bg-white/20 flex items-center justify-center">
          <Icon className="w-5 h-5 text-white" />
        </div>
        <h3 className="text-xl font-bold text-white">{title}</h3>
      </div>
      <div className="space-y-4">
        {items.map((item, index) => (
          <div key={index} className="bg-white/10 rounded-lg p-4 backdrop-blur-sm border border-white/10">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
              <div>
                <span className="font-semibold text-white block">{item.label}</span>
                <span className="text-gray-200 text-sm">{item.value}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 backdrop-blur-3xl"></div>
        <div className="relative container mx-auto px-6 py-20">
          <div className="text-center mb-16">
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-6">
              AI-Powered Analysis
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Transform your images and audio with cutting-edge AI technology. Get detailed insights and analysis in seconds.
            </p>
          </div>

          {/* Feature Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <FeatureCard 
              icon={Eye} 
              title="Facial Analysis" 
              description="Advanced AI analyzes facial features with precision"
              color="from-blue-500 to-blue-600"
            />
            <FeatureCard 
              icon={Music} 
              title="Audio Processing" 
              description="Extract insights from voice and audio patterns"
              color="from-purple-500 to-purple-600"
            />
            <FeatureCard 
              icon={Zap} 
              title="Instant Results" 
              description="Get comprehensive analysis in real-time"
              color="from-green-500 to-green-600"
            />
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 pb-20">
        {/* Image Upload Section */}
        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-sm rounded-2xl p-8 mb-12 border border-gray-700/50 shadow-2xl">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center">
              <ImageIcon className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-white">Image Analysis</h2>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Upload Area */}
            <div className="space-y-6">
              <div 
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                  dragOver.image 
                    ? 'border-blue-400 bg-blue-400/10' 
                    : 'border-gray-600 hover:border-gray-500 bg-gray-800/50'
                }`}
                onDragOver={(e) => handleDragOver(e, 'image')}
                onDragLeave={(e) => handleDragLeave(e, 'image')}
                onDrop={(e) => handleDrop(e, 'image')}
              >
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-semibold text-white mb-2">
                  {selectedImage ? selectedImage.name : "Drop your image here"}
                </p>
                <p className="text-gray-400">or click to browse</p>
              </div>

              {selectedImage && (
                <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur-sm border border-gray-700/50">
                  <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    Image Preview
                  </h3>
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="Selected Preview"
                    className="w-full max-w-sm mx-auto rounded-lg shadow-lg"
                  />
                </div>
              )}

              <button
                onClick={handleSubmit}
                disabled={loading || !selectedImage}
                className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 shadow-lg hover:shadow-xl"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing Image...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Analyze Image
                  </>
                )}
              </button>
            </div>

            {/* Results Area */}
            <div className="space-y-6">
              {error && (
                <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 backdrop-blur-sm">
                  <div className="flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <p className="text-red-300">Error: {error}</p>
                  </div>
                </div>
              )}

              {processedImage && (
                <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur-sm border border-gray-700/50">
                  <h3 className="font-semibold text-white mb-4">Processed Result</h3>
                  <img
                    src={`data:image/png;base64,${processedImage}`}
                    alt="Processed Preview"
                    className="w-full max-w-sm mx-auto rounded-lg shadow-lg"
                  />
                </div>
              )}

              {result && !error && (
                <AnalysisCard
                  title="Facial Features Analysis"
                  icon={Eye}
                  gradient="from-blue-600/20 to-purple-600/20"
                  items={[
                    { label: "Eye Shape", value: formatDescription(result.eye.shape, result.eye.description) },
                    { label: "Eyebrow Shape", value: formatDescription(result.eyebrow.shape, result.eyebrow.description) },
                    { label: "Jaw Shape", value: formatDescription(result.jaw.shape, result.jaw.description) },
                    { label: "Mouth Shape", value: formatDescription(result.mouth.shape, result.mouth.description) },
                    { label: "Nose Shape", value: formatDescription(result.nose.shape, result.nose.description) }
                  ]}
                />
              )}
            </div>
          </div>
        </div>

        {/* Audio Upload Section */}
        <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 shadow-2xl">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-center">
              <Music className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-white">Audio Analysis</h2>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Upload Area */}
            <div className="space-y-6">
              <div 
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                  dragOver.audio 
                    ? 'border-purple-400 bg-purple-400/10' 
                    : 'border-gray-600 hover:border-gray-500 bg-gray-800/50'
                }`}
                onDragOver={(e) => handleDragOver(e, 'audio')}
                onDragLeave={(e) => handleDragLeave(e, 'audio')}
                onDrop={(e) => handleDrop(e, 'audio')}
              >
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleAudioChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Music className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-semibold text-white mb-2">
                  {selectedAudio ? selectedAudio.name : "Drop your audio here"}
                </p>
                <p className="text-gray-400">or click to browse</p>
              </div>

              {audioPreview && (
                <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur-sm border border-gray-700/50">
                  <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    Audio Preview
                  </h3>
                  <div className="bg-gray-700/50 rounded-lg p-4 flex items-center gap-4">
                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="w-10 h-10 rounded-full bg-purple-500 hover:bg-purple-600 flex items-center justify-center transition-colors"
                    >
                      {isPlaying ? <Pause className="w-5 h-5 text-white" /> : <Play className="w-5 h-5 text-white ml-0.5" />}
                    </button>
                    <div className="flex-1">
                      <audio controls src={audioPreview} className="w-full"></audio>
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={handleAudioSubmit}
                disabled={audioLoading || !selectedAudio}
                className="w-full bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 shadow-lg hover:shadow-xl"
              >
                {audioLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing Audio...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Analyze Audio
                  </>
                )}
              </button>
            </div>

            {/* Results Area */}
            <div className="space-y-6">
              {audioError && (
                <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 backdrop-blur-sm">
                  <div className="flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <p className="text-red-300">Error: {audioError}</p>
                  </div>
                </div>
              )}

              {audioResult && !audioError && (
                <AnalysisCard
                  title="Audio Analysis Results"
                  icon={Music}
                  gradient="from-purple-600/20 to-pink-600/20"
                  items={Object.entries(audioResult).map(([trait, score]) => ({
                    label: trait,
                    value: score
                  }))}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageUploadSection;