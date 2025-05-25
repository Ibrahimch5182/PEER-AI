import { brainwaveSymbol, check } from "../assets";
import { collabApps, collabContent, collabText } from "../constants";
import Button from "./Button";
import Section from "./Section";
import { LeftCurve, RightCurve } from "./design/Collaboration";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";

const Collaboration = () => {
  const [contentRef, contentInView] = useInView({
    triggerOnce: true,
    threshold: 0.2,
  });

  const [graphicRef, graphicInView] = useInView({
    triggerOnce: true,
    threshold: 0.2,
  });

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

  const checkmarkVariants = {
    hidden: { scale: 0, opacity: 0 },
    visible: { 
      scale: 1, 
      opacity: 1,
      transition: { type: "spring", stiffness: 300, damping: 15 }
    },
  };

  const graphicVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 1,
      scale: 1,
      transition: { duration: 0.8, ease: "easeOut" }
    },
  };

  const appIconVariants = {
    hidden: { opacity: 0, scale: 0 },
    visible: (i) => ({ 
      opacity: 1, 
      scale: 1, 
      transition: { 
        delay: 0.4 + (i * 0.1),
        type: "spring",
        stiffness: 260,
        damping: 20,
      }
    }),
  };

  const pulseAnimation = {
    scale: [1, 1.05, 1],
    opacity: [0.7, 1, 0.7],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: "easeInOut",
    },
  };

  return (
    <Section crosses className="overflow-hidden">
      <div className="container lg:flex lg:gap-10 xl:gap-20">
        <motion.div 
          ref={contentRef}
          variants={containerVariants}
          initial="hidden"
          animate={contentInView ? "visible" : "hidden"}
          className="max-w-[35rem]"
        >
          <motion.h2 
            variants={itemVariants} 
            className="h2 mb-4 md:mb-8 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-transparent bg-clip-text"
          >
            What is PEER-AI?
          </motion.h2>

          <motion.ul 
            variants={containerVariants}
            className="max-w-[40rem] mb-10 md:mb-14"
          >
            {collabContent.map((item) => (
              <motion.li 
                variants={itemVariants}
                className="mb-3 py-3 border-b border-n-6/40 hover:border-n-6 transition-colors duration-300" 
                key={item.id}
                whileHover={{ x: 10, transition: { duration: 0.2 } }}
              >
                <div className="flex items-center">
                  <motion.div 
                    variants={checkmarkVariants} 
                    className="flex-shrink-0"
                  >
                    <div className="w-6 h-6 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 flex items-center justify-center">
                      <img src={check} width={14} height={14} alt="check" />
                    </div>
                  </motion.div>
                  <h6 className="body-2 ml-5 font-medium">{item.title}</h6>
                </div>
                {item.text && (
                  <motion.p 
                    variants={itemVariants}
                    className="body-2 mt-3 text-n-4 pl-11"
                  >
                    {item.text}
                  </motion.p>
                )}
              </motion.li>
            ))}
          </motion.ul>
         
          <motion.div 
            variants={itemVariants}
            className="mb-10 p-6 rounded-xl bg-n-7/50 backdrop-blur-sm border border-n-6/50"
          >
            <motion.h3 
              variants={itemVariants}
              className="h3 mb-4 bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-500 text-transparent bg-clip-text"
            >
              The possibilities are beyond your imagination
            </motion.h3>
            <motion.p 
              variants={itemVariants}
              className="body-2 text-n-4"
            >
              "Join us for early access to our groundbreaking AI model for advanced personality evaluation, integrating facial recognition and handwriting analysis. Be among the first to leverage cutting-edge technology for precise and efficient personality assessments."
            </motion.p>
          </motion.div>

          <motion.div variants={itemVariants}>
            <Button 
              onClick={() => document.getElementById('image-upload').scrollIntoView({ behavior: 'smooth' })}
              className="group relative overflow-hidden bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500"
            >
              <span className="relative z-10">Try it now</span>
              <span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity duration-300"></span>
            </Button>
          </motion.div>
        </motion.div>

        <motion.div
          ref={graphicRef}
          variants={graphicVariants}
          initial="hidden"
          animate={graphicInView ? "visible" : "hidden"}
          className="lg:ml-auto xl:w-[38rem] mt-4"
        >
          <motion.p 
            variants={itemVariants}
            className="body-2 mb-8 text-n-4 md:mb-16 lg:mb-32 lg:w-[35rem] lg:mx-auto"
          >
            {collabText}
          </motion.p>

          <motion.div 
            animate={pulseAnimation}
            className="relative left-1/2 flex w-[22rem] aspect-square border border-n-6 rounded-full -translate-x-1/2 scale:75 md:scale-100"
          >
            <motion.div 
              animate={{ rotate: 360 }}
              transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
              className="flex w-60 aspect-square m-auto border border-n-6 rounded-full"
            >
              <motion.div 
                whileHover={{ scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
                className="w-[6rem] aspect-square m-auto p-[0.2rem] bg-conic-gradient rounded-full"
              >
                <div className="flex items-center justify-center w-full h-full bg-n-8 rounded-full">
                  <img
                    src={brainwaveSymbol}
                    width={48}
                    height={48}
                    alt="brainwave"
                    className="transform hover:scale-110 transition-transform duration-300"
                  />
                </div>
              </motion.div>
            </motion.div>

            <ul>
              {collabApps.map((app, index) => (
                <motion.li
                  key={app.id}
                  custom={index}
                  variants={appIconVariants}
                  className={`absolute top-0 left-1/2 h-1/2 -ml-[1.6rem] origin-bottom rotate-${
                    index * 45
                  }`}
                >
                  <motion.div
                    whileHover={{ scale: 1.2, boxShadow: "0 0 20px rgba(93, 69, 255, 0.4)" }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                    className={`relative -top-[1.6rem] flex w-[3.2rem] h-[3.2rem] bg-n-7 border border-n-1/15 rounded-xl -rotate-${
                      index * 45
                    } hover:border-indigo-500/50 transition-colors duration-300`}
                  >
                    <img
                      className="m-auto"
                      width={app.width}
                      height={app.height}
                      alt={app.title}
                      src={app.icon}
                    />
                  </motion.div>
                </motion.li>
              ))}
            </ul>

            <LeftCurve />
            <RightCurve />
          </motion.div>
        </motion.div>
      </div>
    </Section>
  );
};

export default Collaboration;