import { useLocation } from "react-router-dom";

const AudioResults = () => {
  const location = useLocation();
  const { audioResult } = location.state || {};

  return (
    <div className="container mx-auto py-16 text-center">
      <h2 className="text-3xl font-bold mb-6">Audio Processing Results</h2>
      {audioResult ? (
        <div>
          <h3 className="text-2xl font-semibold mt-6">Analysis:</h3>
          <p className="mt-4">{JSON.stringify(audioResult)}</p>
        </div>
      ) : (
        <p className="text-red-500 mt-6">No results found. Please try again.</p>
      )}
    </div>
  );
};

export default AudioResults;


