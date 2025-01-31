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
const App = () => {
  return (
    <>
      <div className="pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden">
        <Header />
        <Hero />
        <Collaboration />
        <ImageUpload />
        <Zodiac />
        <Roadmap />
        <Benefits />
        <Footer />
      </div>

      <ButtonGradient />
    </>
  );
};

export default App;
