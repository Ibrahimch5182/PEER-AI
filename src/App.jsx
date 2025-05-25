import { Routes, Route, useLocation } from "react-router-dom";
import ButtonGradient from "./assets/svg/ButtonGradient";
import Benefits from "./components/Benefits";
import Collaboration from "./components/Collaboration";
import Footer from "./components/Footer";
import Header from "./components/Header";
import Hero from "./components/Hero";
import Roadmap from "./components/Roadmap";
import Login from "./components/Login";
import Signup from "./components/Signup";
import LinkedInAnalysis from "./components/LinkedInAnalysis";
import CVUpload from "./components/CVUpload";
import ImageUpload from "./components/ImageUpload";
import FeaturePreview from "./components/FeaturePreview";
import PageTransition from "./PageTransition"; // ✅ your new transition wrapper

const App = () => {
  const location = useLocation();

  return (
    <>
      <div className="pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden">
        <Header />

        {/* Apply transitions to all route content */}
        <PageTransition key={location.pathname}>
          <Routes location={location}>
            {/* Home Page */}
            <Route
              path="/"
              element={
                <>
                  <Hero />
                  <Collaboration />
                  <FeaturePreview />
                  <Roadmap />
                  <Benefits />
                  <Footer />
                </>
              }
            />

            {/* Feature Pages */}
            <Route path="/cv" element={<CVUpload />} />
            <Route path="/linkedin" element={<LinkedInAnalysis />} />
            <Route path="/image" element={<ImageUpload />} />

            {/* Auth Routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
          </Routes>
        </PageTransition>
      </div>

      <ButtonGradient />
    </>
  );
};

export default App;
