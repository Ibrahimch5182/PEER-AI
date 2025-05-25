import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import React from "react";

const FeaturePreview = () => {
  const navigate = useNavigate();
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { duration: 0.8, ease: "easeOut" }
    },
  };

  const features = [
    { 
      name: "CV Analyzer", 
      path: "/cv", 
      description: "Upload your resume to get comprehensive personality insights based on your professional background and achievements.",
      icon: "📄",
      color: "from-blue-600 to-cyan-400" 
    },
    { 
      name: "LinkedIn Analyzer", 
      path: "/linkedin", 
      description: "Analyze LinkedIn profiles to reveal personality traits, professional tendencies, and potential career paths.",
      icon: "🔗",
      color: "from-indigo-600 to-blue-400" 
    },
    { 
      name: "Image & Audio Analysis", 
      path: "/image", 
      description: "Discover hidden personality insights through facial recognition and voice pattern analysis technology.",
      icon: "🎭",
      color: "from-purple-600 to-pink-400" 
    },
    { 
      name: "HR Candidate Screening", 
      path: "https://www.klarushr.com/", 
      description: "Advanced AI-powered recruitment platform for intelligent candidate screening and personality assessment in hiring processes.",
      icon: "🎯",
      color: "from-emerald-600 to-green-400",
      isExternal: true
    },
  ];

  return (
    <motion.div
      ref={ref}
      variants={containerVariants}
      initial="hidden"
      animate={inView ? "visible" : "hidden"}
      className="py-16 px-4 md:py-24"
    >
      <div className="max-w-6xl mx-auto">
        <motion.div 
          variants={itemVariants}
          className="text-center mb-12"
        >
          <div className="inline-block py-1 px-3 rounded-full bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 backdrop-blur-sm mb-4">
            <span className="text-sm text-indigo-300 font-medium">Comprehensive Analysis</span>
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-4 bg-gradient-to-r from-indigo-400 via-purple-500 to-pink-400 text-transparent bg-clip-text">
            Analyze From Multiple Perspectives
          </h2>
          <p className="text-lg text-n-3 max-w-2xl mx-auto">
            Our AI models evaluate personality traits through different data points to provide a holistic understanding.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mt-10">
          {features.map((feature, index) => (
            <motion.div
              key={feature.name}
              variants={itemVariants}
              className="relative group"
            >
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r opacity-70 blur-lg group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative h-full bg-n-7 rounded-2xl overflow-hidden border border-n-5 shadow-lg transform group-hover:scale-[1.01] transition-transform duration-300">
                {/* Top gradient bar */}
                <div className={`h-2 w-full bg-gradient-to-r ${feature.color}`}></div>
                
                <div className="p-6 md:p-8 flex flex-col h-full">
                  {/* Icon and Title */}
                  <div className="flex items-center mb-4">
                    <div className={`flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-br ${feature.color} text-white text-xl mb-4`}>
                      {feature.icon}
                    </div>
                    <h3 className="text-xl font-semibold ml-4">{feature.name}</h3>
                  </div>
                  
                  {/* Description */}
                  <p className="text-n-3 text-sm mb-6 flex-grow">{feature.description}</p>
                  
                  {/* Feature preview graphic - simplified visualization */}
                  <div className="w-full h-32 mb-6 rounded-lg bg-n-8 overflow-hidden relative">
                    <div className={`absolute inset-0 bg-gradient-to-r ${feature.color} opacity-5`}></div>
                    
                    {/* Simplified data visualization elements */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      {feature.name === "CV Analyzer" && (
                        <div className="w-full px-4">
                          <div className="flex justify-between mb-2">
                            <span className="text-xs text-n-3">Openness</span>
                            <span className="text-xs text-n-3">75%</span>
                          </div>
                          <div className="w-full bg-n-6 rounded-full h-1.5">
                            <motion.div 
                              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400"
                              initial={{ width: 0 }}
                              animate={{ width: "75%" }}
                              transition={{ duration: 1, delay: 0.5 + index * 0.2 }}
                            ></motion.div>
                          </div>
                          <div className="flex justify-between mt-3 mb-2">
                            <span className="text-xs text-n-3">Conscientiousness</span>
                            <span className="text-xs text-n-3">82%</span>
                          </div>
                          <div className="w-full bg-n-6 rounded-full h-1.5">
                            <motion.div 
                              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400"
                              initial={{ width: 0 }}
                              animate={{ width: "82%" }}
                              transition={{ duration: 1, delay: 0.7 + index * 0.2 }}
                            ></motion.div>
                          </div>
                        </div>
                      )}
                      
                      {feature.name === "LinkedIn Analyzer" && (
                        <div className="w-full h-full flex items-center justify-center">
                          <svg className="w-full h-full" viewBox="0 0 200 100">
                            <motion.path
                              d="M 10 70 Q 40 20 70 50 Q 100 80 130 30 Q 160 10 190 40"
                              fill="none"
                              stroke="url(#blue-gradient)"
                              strokeWidth="2"
                              initial={{ pathLength: 0, opacity: 0 }}
                              animate={{ pathLength: 1, opacity: 1 }}
                              transition={{ duration: 2, delay: 0.5 }}
                            />
                            <defs>
                              <linearGradient id="blue-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#2563EB" />
                                <stop offset="100%" stopColor="#22D3EE" />
                              </linearGradient>
                            </defs>
                          </svg>
                        </div>
                      )}
                      
                      {feature.name === "Image & Audio Analysis" && (
                        <div className="w-full h-full flex">
                          <div className="w-1/2 border-r border-n-6 flex items-center justify-center">
                            <motion.div 
                              className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-500 to-pink-400 opacity-50"
                              animate={{ 
                                scale: [1, 1.1, 1],
                                opacity: [0.5, 0.7, 0.5]
                              }}
                              transition={{ duration: 2, repeat: Infinity }}
                            ></motion.div>
                          </div>
                          <div className="w-1/2 flex items-center justify-center">
                            <svg className="w-full h-10" viewBox="0 0 100 40">
                              <motion.path
                                d="M 0 20 Q 5 10 10 20 Q 15 30 20 20 Q 25 10 30 20 Q 35 30 40 20 Q 45 10 50 20 Q 55 30 60 20 Q 65 10 70 20 Q 75 30 80 20 Q 85 10 90 20 Q 95 30 100 20"
                                fill="none"
                                stroke="url(#purple-gradient)"
                                strokeWidth="2"
                                initial={{ pathLength: 0, opacity: 0 }}
                                animate={{ pathLength: 1, opacity: 1 }}
                                transition={{ duration: 2, delay: 0.5 }}
                              />
                              <defs>
                                <linearGradient id="purple-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                  <stop offset="0%" stopColor="#A855F7" />
                                  <stop offset="100%" stopColor="#EC4899" />
                                </linearGradient>
                              </defs>
                            </svg>
                          </div>
                        </div>
                      )}

                      {feature.name === "HR Candidate Screening" && (
                        <div className="w-full h-full flex items-center justify-center">
                          <div className="w-full px-4">
                            <div className="grid grid-cols-3 gap-2 mb-3">
                              <motion.div 
                                className="h-8 bg-gradient-to-r from-emerald-500 to-green-400 rounded opacity-40"
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                              ></motion.div>
                              <motion.div 
                                className="h-8 bg-gradient-to-r from-emerald-500 to-green-400 rounded opacity-60"
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ duration: 0.5, delay: 0.7 + index * 0.1 }}
                              ></motion.div>
                              <motion.div 
                                className="h-8 bg-gradient-to-r from-emerald-500 to-green-400 rounded opacity-80"
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ duration: 0.5, delay: 0.9 + index * 0.1 }}
                              ></motion.div>
                            </div>
                            <div className="text-center">
                              <div className="text-xs text-n-3">Candidate Match</div>
                              <motion.div 
                                className="text-lg font-bold text-emerald-400"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 1, delay: 1.2 }}
                              >
                                92%
                              </motion.div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Button */}
                  <motion.button
                    className={`w-full py-3 px-4 rounded-lg bg-gradient-to-r ${feature.color} text-white font-medium transition-all duration-300 overflow-hidden relative`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      if (feature.isExternal) {
                        window.open(feature.path, '_blank');
                      } else {
                        navigate(feature.path);
                      }
                    }}
                  >
                    <span className="relative z-10 flex items-center justify-center">
                      {feature.isExternal ? 'Visit' : 'Use'} {feature.name}
                      <svg className="w-5 h-5 ml-2" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        {feature.isExternal ? (
                          <path d="M18 13V19C18 19.5304 17.7893 20.0391 17.4142 20.4142C17.0391 20.7893 16.5304 21 16 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V8C3 7.46957 3.21071 6.96086 3.58579 6.58579C3.96086 6.21071 4.46957 6 5 6H11M15 3H21V9M10 14L21 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        ) : (
                          <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        )}
                      </svg>
                    </span>
                    <motion.div
                      className="absolute inset-0 bg-white opacity-0"
                      whileHover={{ opacity: 0.2 }}
                      transition={{ duration: 0.3 }}
                    />
                  </motion.button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export default FeaturePreview;