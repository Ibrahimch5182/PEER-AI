import { Routes, Route } from "react-router-dom";
import ButtonGradient from "./assets/svg/ButtonGradient";
import Benefits from "./components/Benefits";
import Collaboration from "./components/Collaboration";
import Footer from "./components/Footer";
import Header from "./components/Header";
import Hero from "./components/Hero";
import Pricing from "./components/Pricing";
import Roadmap from "./components/Roadmap";
import Services from "./components/Services";
import ImageUpload from "./components/ImageUpload";
import Zodiac from "./components/design/zodiac";
import Individual from "./components/design/individual";
import Login from "./components/Login";
import Signup from "./components/Signup";
import LinkedInAnalysis from './components/LinkedInAnalysis';
import CVUpload from './components/CVUpload';


const App = () => {
  return (
    <>
      <div className="pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden">
        <Header />
        <Routes>
          {/* Default route */}
          <Route path="/" element={
            <>
              <Hero />
              <Collaboration />
              <ImageUpload />
              <Zodiac />
              <LinkedInAnalysis />
              <CVUpload />
              <Roadmap />
              <Benefits />
              <Services />
              <Pricing />
              <Footer />
            </>
          } />

          {/* Login route */}
          <Route path="/login" element={<Login />} />

          {/* Signup route */}
          <Route path="/signup" element={<Signup />} />
        </Routes>
      </div>

      <ButtonGradient />
    </>
  );
};

export default App;