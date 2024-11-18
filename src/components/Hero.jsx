import Section from "./Section"; 
import Spline from "@splinetool/react-spline"; 

const Hero = () => {
  return (
    <Section
      className="pt-[12rem] -mt-[5.25rem]"
      customPaddings
      id="hero"
    >
      <div className="container relative">
        <Spline scene="https://prod.spline.design/eDAMtYmUD9T1t1Bo/scene.splinecode" />
      </div>
    </Section>
  );
};

export default Hero;
