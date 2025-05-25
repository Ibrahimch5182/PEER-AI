import React, { useEffect } from "react";
import Section from "./Section";
import { motion, useAnimation } from "framer-motion";
import { useNavigate } from "react-router-dom";

const Hero = () => {
  const navigate = useNavigate();
  const controls = useAnimation();

  useEffect(() => {
    controls.start("visible");
  }, [controls]);

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.6, ease: "easeOut" }
    },
  };

  const glowEffectVariants = {
    initial: { scale: 0.95, opacity: 0 },
    animate: { 
      scale: 1.05, 
      opacity: [0.2, 0.4, 0.2], 
      transition: { 
        duration: 4,
        repeat: Infinity,
        repeatType: "reverse"
      }
    }
  };

  const orbitalCircleVariants = {
    initial: { rotate: 0 },
    animate: { 
      rotate: 360,
      transition: { 
        duration: 20, 
        repeat: Infinity, 
        ease: "linear" 
      }
    }
  };

  // Data objects for animated brain activity nodes
  const brainNodes = [
    { x: "20%", y: "30%", size: 6, delay: 0 },
    { x: "55%", y: "15%", size: 10, delay: 0.5 },
    { x: "80%", y: "40%", size: 8, delay: 1 },
    { x: "35%", y: "70%", size: 12, delay: 1.5 },
    { x: "65%", y: "60%", size: 7, delay: 2 },
    { x: "90%", y: "80%", size: 9, delay: 2.5 },
    { x: "10%", y: "55%", size: 8, delay: 3 },
    { x: "50%", y: "85%", size: 11, delay: 3.5 },
  ];

  return (
    <Section className="pt-[12rem] -mt-[5.25rem] overflow-hidden" customPaddings id="hero">
      <div className="container relative z-10">
        {/* Main content */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={controls}
          className="relative z-10 flex flex-col items-center text-center px-4"
        >
          <motion.div
            variants={itemVariants}
            className="inline-block py-1 px-4 mb-4 rounded-full bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 backdrop-blur-sm"
          >
            <span className="text-sm text-indigo-300 font-medium">Next-Gen AI Personality Analysis</span>
          </motion.div>
          
          <motion.h1 
            variants={itemVariants} 
            className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-indigo-400 via-purple-500 to-pink-400 text-transparent bg-clip-text"
          >
            Meet Peer-AI
          </motion.h1>
          
          <motion.p 
            variants={itemVariants}
            className="text-xl md:text-2xl text-n-1/80 max-w-2xl mb-8 leading-relaxed"
          >
            Pre Evaluated Efficiency Resource - Advanced AI-powered analysis
            for comprehensive personality insights through multimodal evaluation.
          </motion.p>
          
          <motion.div 
            variants={itemVariants}
            className="flex flex-col md:flex-row gap-4 md:gap-6"
          >
            <motion.button 
              className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full text-white font-medium flex items-center justify-center gap-2 group"
              whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(93, 69, 255, 0.4)" }}
              whileTap={{ scale: 0.95 }}
              onClick={() => document.getElementById('image-upload').scrollIntoView({ behavior: 'smooth' })}
            >
              <span>Try Now</span>
              <span className="transform group-hover:translate-x-1 transition-transform duration-300">→</span>
            </motion.button>
            
            <motion.button 
              className="px-8 py-3 border border-indigo-500/30 bg-n-8/50 backdrop-blur-sm rounded-full text-white font-medium"
              whileHover={{ 
                backgroundColor: "rgba(255, 255, 255, 0.1)", 
                borderColor: "rgba(129, 93, 255, 0.5)" 
              }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/features")}
            >
              Explore Features
            </motion.button>
          </motion.div>
        </motion.div>

        {/* Abstract AI Brain Visualization */}
        <div className="relative mt-16 md:mt-20 h-[40vh] md:h-[50vh] w-full max-w-5xl mx-auto">
          {/* Central "brain" sphere */}
          <motion.div 
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-28 h-28 md:w-40 md:h-40 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 opacity-70 blur-sm"
            animate={{
              scale: [1, 1.05, 1],
              opacity: [0.7, 0.8, 0.7],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />

          {/* Glow effect behind the brain */}
          <motion.div 
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-40 h-40 md:w-64 md:h-64 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 opacity-20 blur-xl"
            variants={glowEffectVariants}
            initial="initial"
            animate="animate"
          />

          {/* Orbital circle */}
          <motion.div 
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-48 h-48 md:w-80 md:h-80 rounded-full border border-indigo-500/20"
            variants={orbitalCircleVariants}
            initial="initial"
            animate="animate"
          />

          {/* Outer orbital circle */}
          <motion.div 
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 md:w-96 md:h-96 rounded-full border border-purple-500/10"
            variants={orbitalCircleVariants}
            initial="initial"
            animate={{ 
              rotate: -360,
              transition: { 
                duration: 25, 
                repeat: Infinity, 
                ease: "linear" 
              }
            }}
          />

          {/* Brain activity nodes */}
          {brainNodes.map((node, index) => (
            <motion.div
              key={index}
              className="absolute rounded-full bg-gradient-to-r from-indigo-400 to-purple-400"
              style={{
                width: `${node.size}px`,
                height: `${node.size}px`,
                left: node.x,
                top: node.y,
              }}
              initial={{ opacity: 0, scale: 0 }}
              animate={{
                opacity: [0, 0.8, 0],
                scale: [0, 1, 0],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                delay: node.delay,
                repeatDelay: Math.random() * 2,
              }}
            />
          ))}

          {/* Connection lines between nodes - animated gradient lines */}
          <svg className="absolute inset-0 w-full h-full opacity-30" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="line-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#6366F1" />
                <stop offset="100%" stopColor="#A855F7" />
              </linearGradient>
            </defs>
            
            {/* Random connection paths between nodes */}
            <motion.path 
              d="M 20% 30% L 55% 15% L 80% 40% L 65% 60% Z" 
              stroke="url(#line-gradient)" 
              strokeWidth="1" 
              fill="none" 
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.3 }}
              transition={{ duration: 2, repeat: Infinity, repeatType: "loop", repeatDelay: 1 }}
            />
            <motion.path 
              d="M 35% 70% L 55% 15% L 65% 60% L 10% 55% Z" 
              stroke="url(#line-gradient)" 
              strokeWidth="1" 
              fill="none"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.3 }}
              transition={{ duration: 2.5, repeat: Infinity, repeatType: "loop", repeatDelay: 0.5 }}
            />
            <motion.path 
              d="M 90% 80% L 50% 85% L 20% 30% L 80% 40% Z" 
              stroke="url(#line-gradient)" 
              strokeWidth="1" 
              fill="none"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.3 }}
              transition={{ duration: 3, repeat: Infinity, repeatType: "loop", repeatDelay: 1.5 }}
            />
          </svg>

          {/* Floating particles */}
          {Array.from({ length: 20 }).map((_, index) => (
            <motion.div
              key={`particle-${index}`}
              className="absolute rounded-full bg-gradient-to-r from-indigo-500/20 to-purple-500/20"
              style={{
                width: Math.random() * 6 + 2,
                height: Math.random() * 6 + 2,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [Math.random() * 50, Math.random() * -50],
                x: [Math.random() * 50, Math.random() * -50],
                opacity: [0, 0.7, 0],
              }}
              transition={{
                duration: Math.random() * 10 + 5,
                repeat: Infinity,
                repeatType: "reverse",
                ease: "easeInOut",
                delay: Math.random() * 5,
              }}
            />
          ))}
        </div>

        {/* Stats section below visualization */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={controls}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto text-center"
        >
          {[
            { label: "Accuracy", value: "98%" },
            { label: "Features", value: "4+" },
            { label: "Models", value: "5" },
            { label: "Data Points", value: "1M+" }
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              variants={itemVariants}
              className="p-4 rounded-xl bg-gradient-to-b from-n-7/50 to-n-8/50 border border-n-6/50 backdrop-blur-sm"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 + index * 0.2, duration: 0.5 }}
                className="text-3xl md:text-4xl font-bold mb-1 bg-gradient-to-r from-indigo-400 to-purple-400 text-transparent bg-clip-text"
              >
                {stat.value}
              </motion.div>
              <div className="text-sm text-n-3">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Animated background gradient */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-0 left-0 right-0 h-[70%] bg-gradient-to-b from-indigo-900/20 to-transparent opacity-30 blur-3xl"></div>
          <div className="absolute bottom-0 left-0 right-0 h-[30%] bg-gradient-to-t from-indigo-900/10 to-transparent opacity-20 blur-2xl"></div>
          <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 bg-gradient-to-tr from-purple-700/10 via-indigo-700/10 to-blue-700/10 opacity-30 blur-3xl rounded-full"></div>
        </div>
      </div>
    </Section>
  );
};

export default Hero;