import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';

const PageTransition = ({ children }) => {
  const location = useLocation();
  
  // Different transition variants
  const transitions = {
    fade: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 },
      transition: { duration: 0.5 }
    },
    slide: {
      initial: { x: '100%' },
      animate: { x: 0 },
      exit: { x: '-100%' },
      transition: { duration: 0.5, ease: "easeInOut" }
    },
    scale: {
      initial: { opacity: 0, scale: 0.9 },
      animate: { opacity: 1, scale: 1 },
      exit: { opacity: 0, scale: 1.1 },
      transition: { duration: 0.5 }
    },
    flip: {
      initial: { opacity: 0, rotateY: 90 },
      animate: { opacity: 1, rotateY: 0 },
      exit: { opacity: 0, rotateY: -90 },
      transition: { duration: 0.5 }
    },
    slideUp: {
      initial: { y: '100%', opacity: 0 },
      animate: { y: 0, opacity: 1 },
      exit: { y: '-100%', opacity: 0 },
      transition: { duration: 0.5, ease: "easeInOut" }
    }
  };
  
  // Determine which transition to use based on path or some other logic
  const getTransition = (pathname) => {
    // You can customize this logic based on your routes
    if (pathname === '/') return transitions.fade;
    if (pathname.includes('login') || pathname.includes('signup')) return transitions.slideUp;
    if (pathname.includes('profile')) return transitions.scale;
    if (pathname.includes('analytics')) return transitions.flip;
    return transitions.slide; // default transition
  };
  
  const currentTransition = getTransition(location.pathname);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={currentTransition.initial}
        animate={currentTransition.animate}
        exit={currentTransition.exit}
        transition={currentTransition.transition}
        className="min-h-screen"
      >
        {children}
        
        {/* Optional overlay for more dramatic transitions */}
        <motion.div
          className="fixed inset-0 z-50 pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0 }}
          exit={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
        >
          <div className="absolute inset-0 bg-black opacity-30" />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default PageTransition;