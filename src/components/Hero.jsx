import React, { Suspense, useState, useEffect } from "react";
import Section from "./Section";

const LazySpline = React.lazy(() =>
  import("@splinetool/react-spline")
);

const Hero = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.1 }
    );

    const heroSection = document.getElementById("hero");
    if (heroSection) observer.observe(heroSection);

    return () => {
      if (heroSection) observer.unobserve(heroSection);
    };
  }, []);

  return (
    <Section className="pt-[12rem] -mt-[5.25rem]" customPaddings id="hero">
      <div className="container relative">
        {isVisible && (
          <Suspense fallback={<div>Loading...</div>}>
            <LazySpline scene="https://prod.spline.design/eDAMtYmUD9T1t1Bo/scene.splinecode" />
          </Suspense>
        )}
      </div>
    </Section>
  );
};

export default Hero;
